# AllGather：全对全切片

每个 rank 发布自己的 slice，每个 rank 最终得到所有 slice 的按 rank 顺序拼接——两相 all-reduce 的 all-gather 一半——然后内置原语一次调用完成。

> **前置：** [17-broadcast](17-broadcast.md)。任意 ≥ 2 个设备（示例使用 2 和 4 个 sim 设备）。

**建议阅读顺序：** 01 → … → 12 → **13** —— 本页是步骤 13。

## 思路

Allgather 反转了 broadcast 的不对称性：每个 rank 既是生产者也是消费者。每个 rank 贡献一个 slice（`N/P` 个元素），每个 rank 最终得到**按 rank 顺序的拼接** `[x[0], x[1], …, x[P-1]]`。

| 方面 | AllGather |
| ---- | --------- |
| 输入 | 每个 rank 的 slice |
| 输出 | 所有 slice 的拼接，位于**每个** rank |
| 模式 | staging 自己的 slice → barrier → 读取每个对端的 slice |
| 开销 | 每个 rank 向每个对端发送 `N/P`：收到 `(P-1)/P · N` |

你以前见过这个模式：步骤 09 的两相 all-reduce 就是 reduce-scatter **后接 allgather**。本步单独构建 allgather 一半；步骤 14 构建 reduce-scatter 一半。

## 运行

```bash
# 手工：staging、barrier、remote_load 每个对端。
python examples/distributed/13_allgather.py -p a2a3sim -d 0,1

# 揭示：一次调用完成 pld.tensor.allgather。
python examples/distributed/13_allgather.py -p a2a3sim -d 0,1 --mode builtin

# 同一源码在 P=4：
python examples/distributed/13_allgather.py -p a2a3sim -d 0,1,2,3
python examples/distributed/13_allgather.py -p a2a3sim -d 0,1,2,3 --mode builtin
```

预期输出：

```text
OK
```

golden 是按 rank 顺序的拼接——在每个 rank 上完全相同——因此任何产生*错误顺序*（或自己 slice）的 rank 都会失败。

## 走读

两种模式共享一个 `[nr, SIZE]` window：每个 rank 最终都 staging 在自己的行上，并读回每一行——在手工内核里，staging 由调用方显式完成；而在下面的揭示中，push 发生在调用内部。手工内核：

```python
@pl.function(type=pl.FunctionType.InCore)
def hand_step(self, x, y, data, signal):
    ctx = pld.get_comm_ctx(data)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)

    # Phase 1 — stage this rank's slice into its own row.
    local = pl.load(x, [0, 0], [1, SIZE])
    data = pl.store(local, [my_rank, 0], data)

    # Phase 2 — barrier: notify every peer, wait on every peer slot.
    for peer in pl.range(nranks):
        if peer != my_rank:
            pld.system.notify(signal, peer=peer, offsets=[my_rank, 0],
                              value=1, op=pld.NotifyOp.AtomicAdd)
    for src in pl.range(nranks):
        if src != my_rank:
            pld.system.wait(signal, offsets=[src, 0], expected=1,
                            cmp=pld.WaitCmp.Ge)

    # Phase 3 — gather: pull every peer's row into the output.
    for peer in pl.range(nranks):
        recv = pld.tile.remote_load(data, peer=peer, offsets=[peer, 0], shape=[1, SIZE])
        y = pl.store(recv, [0, peer * SIZE], y)
    return y
```

- **行 `my_rank` 是你的槽位。** 在 `[my_rank, 0]` 而不是 broadcast 的单一根槽位 staging，这正是交换对称的原因：每个 rank 写不同的行，因此没有两个 rank 会冲突。
- **gather 是对对端的循环**——对每个对端 `p` 在行 `p` 处 `remote_load`，存入输出偏移 `p * SIZE`。输出是按 rank 顺序的拼接，这正是循环顺序（和偏移计算）重要的原因：槽位 `p` 必须保存 rank `p` 的 slice。

揭示用一次调用替换阶段 2–3——push 形式：

```python
    data = pld.tensor.allgather(x, data, signal)   # stage + barrier + gather

    for src in pl.range(nranks):
        chunk = pl.load(data, [src, 0], [1, SIZE])
        y = pl.store(chunk, [0, src * SIZE], y)
```

- **源是你的本地 `x`（普通 `pl.Tensor`），不是 window。** push 形式的 allgather 替你 staging；目标 window 变成 `[nr, SIZE]` 结果（行 `src` = rank `src` 的 slice）。
- **行布局与手工版本相同，方向相反。** 行 `src` 仍然是 rank `src` 的 slice，因此下面的读取循环无需改动——但内置原语是 *push*，而你的版本是 *pull*，这也让 barrier 换了位置。参见 IR 对比。

### IR 对比（教学工件）

这是本阶梯中第一个 IR 对比真正有看头的步骤：两种模式把相同的字节送进相同的布局，但**方向相反**，barrier 也落在传输的另一侧。

- `--mode hand` lowering 为上述三个阶段，是一次 **pull**：一次写入你的行、notify/wait barrier，然后 `P` 次 `pld.tile.remote_load`——每个对端一次；注意 gather 循环并*不*跳过自己那一行，因此你自己的 slice 也走同一条路径读回。barrier 在数据搬运*之前*，因为你不能读取尚未 staging 的对端。
- `--mode builtin` lowering 为一次 **push**：一个 `P` 次 `pld.tile.put` 的循环，每次把本 rank 的 `[1, SIZE]` chunk 写入该对端 window 的第 `my_rank` 行，*然后*才是 `[nr, 1]` 信号上的 notify/wait barrier。展开结果中根本没有 `remote_load`——上面代码片段里的 `pl.load` 只是对已被各对端填好的 window 做本地读取。barrier 在传输*之后*，因为在这里它的作用是告诉你每个对端的 push 都已落地。
- push 有两个细节值得在 IR 里看到：每次 `pld.tile.put` 都经由共享的 VEC staging tile（`tile.create`）流式传输，因此比该 tile 更大的行会被 pto-isa 自动分块；而自身 rank 的那次迭代（`peer == my_rank`）并未特判——它通过 HCCL 恒等映射走同一条 TPUT 路径。
- barrier 之后 composite 会发出**自清理尾声**，把本次调用的信用额度再减回去——这正是同一个信号可以被后续集合通信复用的原因（见
[21-putting_it_together](21-putting_it_together.md)，步骤 16）。

**成本卡（每 rank）：** 每个 rank 向每个对端发送 `N/P` 字节，因此每个 rank 收到 `(P-1)/P · N` 字节——两相 all-reduce（步骤 09）的 gather 一半，其每一相也都移动 `(P-1)/P · N`。

## 边界情况

> **致命陷阱——把数据 gather 到错误的槽位。** 如果输出偏移与对端 rank 不匹配（从 peer `p` 读出的 `y[p]` 却按 `p+1` 偏移），每个 rank 内部自洽但 golden 仍然失败——顺序就是契约。**修复：** peer `peer` 对应偏移 `peer * SIZE`。
>
> **致命陷阱——同一个 window 既作源又作结果。** `target` 会被重绑为结果 window，
> 但 `local_data` 必须是另一块独立 buffer——这与 `all_to_all` 施加的约束相同
> （步骤 15）。把同一个 window 同时用作两者是跨进程数据竞争：对端向 `target` 的
> push 可能在你仍以 `local_data` 读取它时落地。**修复：** 像示例那样让源与目标
> 保持相互独立——内置实现也会强制这一点。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| 行顺序错误 | 输出偏移 ≠ 对端 rank | 把 peer `p` 存入 `[0, p * SIZE]` |
| 每个 rank 显示自己的 slice | 读取自己的 window 而非对端 | 对每个 `peer` 在 `[peer, 0]` 处 `remote_load` |
| `pld.tensor.allgather local_data must be a plain Tensor` | InCore 路径上源传入了 `DistributedTensor` window | 源必须是与 `target` 不同的普通 `pl.Tensor [1, SIZE]`；window 形式仅 HOST 路径接受 |
| `pld.tensor.allgather input must be a Tensor or DistributedTensor` | 源传入的是 tile——更早一步就被类型推导拒绝 | 直接传张量本身，而不是它的 `pl.load` 结果 |
| `pld.tensor.allgather input and target must be different buffers` | 同一个 window 同时作为 `local_data` 和 `target` 传入 | 另行分配一块独立的源 buffer |
| 拼接有缺口/重叠 | stage/gather 偏移不匹配 | stage 行 `my_rank`；读行 `peer` |
| P=4 时数据陈旧 | stage 与 gather 之间缺少 barrier | 读取循环前 notify/wait 覆盖全部 `nr` 个对端 |

## 另请参阅

- [05-tutorials](05-tutorials.md) — 教程索引（本步 = 第 13 行）
- [01-collectives](../distributed/01-collectives.md) §AllGather — 完整 API
- [14-allreduce_two_phase](14-allreduce_two_phase.md) — 本步隔离出的两相 all-reduce 的 gather 一半
- [04-debugging](04-debugging.md) — 分布式程序的规范故障目录
- 下一步：[19-reduce_scatter](19-reduce_scatter.md) — 归约一半
