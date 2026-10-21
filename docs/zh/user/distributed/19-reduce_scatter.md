# ReduceScatter：全对分块

每个 rank staging 全部分块；每个 rank 最终得到自己索引处的归约分块——两相 all-reduce 的 reduce-scatter 一半——然后内置原语一次调用完成。

> **前置：** [18-allgather](18-allgather.md)。任意 ≥ 2 个设备（示例使用 2 和 4 个 sim 设备）。

**建议阅读顺序：** 01 → … → 13 → **14** —— 本页是步骤 14。

## 思路

Reduce-scatter 是 allgather 的镜像：不是每个 rank 收到*所有* slice，而是每个 rank 收到**一个归约后的分块**——自己索引处的分块，跨所有 rank 归约（这里为求和）。

| 方面 | ReduceScatter |
| ---- | ------------- |
| 输入 | 每个 rank 的完整 `P` 个分块（`N` 个元素） |
| 输出 | rank `r` 得到 `Σ_k chunk_r(inputs[k])`——`N/P` 个元素 |
| 模式 | staging 全部分块 → barrier → 跨对端求和自己的分块 |
| 开销 | 收到 `(P-1)/P · N` 字节——两相的第一半 |

这正是**步骤 09 两相 all-reduce 的第一半**。步骤 13 构建了第二半（allgather）；本步构建第一半。合起来就是你已经作为单个内置原语运行过的两相调度。

## 运行

```bash
# 手工：staging 全部分块、barrier、跨对端求和自己的分块。
python examples/distributed/14_reduce_scatter.py -p a2a3sim -d 0,1

# 揭示：一次调用完成 pld.tensor.reduce_scatter。
python examples/distributed/14_reduce_scatter.py -p a2a3sim -d 0,1 --mode builtin

# 同一源码在 P=4：
python examples/distributed/14_reduce_scatter.py -p a2a3sim -d 0,1,2,3
python examples/distributed/14_reduce_scatter.py -p a2a3sim -d 0,1,2,3 --mode builtin
```

预期输出：

```text
OK
```

golden 是逐 rank 的：`out[r]` 必须等于跨所有 rank 的分块 `r` 的逐元素和——每个 rank 是*不同*的分块，因此归约错分块的 rank 会失败。

## 走读

两种模式共享一个 `[nr, SIZE]` window：每个 rank 把分块 `c` staging 到行 `c`，并归约行 `my_rank`。手工内核：

```python
@pl.function(type=pl.FunctionType.InCore)
def hand_step(self, x, y, data, signal):
    ctx = pld.get_comm_ctx(data)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)

    # Phase 1 — stage every chunk at its row, so each peer can read it.
    for c in pl.range(nranks):
        chunk = pl.load(x, [0, c * SIZE], [1, SIZE])
        data = pl.store(chunk, [c, 0], data)

    # Phase 2 — barrier: notify every peer, wait on every peer slot.
    for peer in pl.range(nranks):
        if peer != my_rank:
            pld.system.notify(signal, peer=peer, offsets=[my_rank, 0],
                              value=1, op=pld.NotifyOp.AtomicAdd)
    for src in pl.range(nranks):
        if src != my_rank:
            pld.system.wait(signal, offsets=[src, 0], expected=1,
                            cmp=pld.WaitCmp.Ge)

    # Phase 3 — reduce: sum row my_rank across every peer.
    acc = pl.load(data, [my_rank, 0], [1, SIZE])
    for peer in pl.range(nranks):
        if peer != my_rank:
            recv = pld.tile.remote_load(data, peer=peer, offsets=[my_rank, 0], shape=[1, SIZE])
            acc = pl.add(acc, recv)
    return pl.store(acc, [0, 0], y)
```

- **你 staging 全部分块，而不仅仅是自己的。** 每个 rank 发布整个 `[nr, SIZE]` 矩阵，这样任何对端都能读取它需要的具体分块。这与 allgather 的单 slice staging 相反——你发布的数据是 `P` 个 slice，消费的数据是一个。
- **归约是本地循环。** remote 读取是*加法*而不是存储：`acc` 从每个对端累加分块 `my_rank`。循环顺序在不同 rank 间不同，这正是 golden 使用容差的原因（归约顺序与 torch 不同）。

揭示用一次调用替换阶段 2–3：

```python
    for c in pl.range(nranks):
        chunk = pl.load(x, [0, c * SIZE], [1, SIZE])
        data = pl.store(chunk, [c, 0], data)

    data = pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)
    acc = pl.load(data, [my_rank, 0], [1, SIZE])
    return pl.store(acc, [0, 0], y)
```

- **`op=` 选择归约方式——这条路径上支持完整家族。** 该参数默认为
  `pld.ReduceOp.Sum`；在本示例使用的 InCore 复合路径上，`Max`、`Min`、`Prod`
  同样受支持，与 `pld.tensor.allreduce` 使用的是同一套分派。限制在别处更窄：
  **HOST 内置**路径（另一种 lowering，本步骤不会走到）只降级 `Sum`/FP32。
  非 Sum 的 op 会以
  `builtin.tensor.reduce_scatter op must be ReduceOp.Sum (got int N)` 拒绝；
  `Sum` 配合非 FP32 的 dtype 则会以
  `builtin.tensor.reduce_scatter currently supports only (op=ReduceOp.Sum, dtype=FP32)`
  拒绝。不要把这条仅限 HOST 的限制带到你在这里写的 InCore 形态上。
- **window 的行 `my_rank` 就是你的归约分块**——与手工版本相同的行-分块布局。

### IR 对比（教学工件）

- `--mode hand` lowering 为上述四个阶段：`P` 次存储、就绪 barrier、以及用加法累加的 `P-1` 次 remote load——随后结果直接写出到 `y`。
- `--mode builtin` 展开为该形状，**外加一个你没有写的第二个 barrier**。因为 composite 把归约后的 chunk 写回 `target[my_rank]` 而不是写出到 `y`，它带有你的版本所没有的写后读（WAR）冒险：快的 rank 可能在慢的对端还在读自己那一行时就将其覆盖。因此展开结果是 就绪 barrier → 归约 → **归约后 barrier** → 向第 `my_rank` 行 `tile.store`，信号每次调用消耗 **2 份信用额度**而非 1 份。
- 这正是本次对比的要点：多出的 barrier 不是编译器没能优化掉的开销——它是原地 window-as-result 形式的代价。你的版本之所以不需要它，只是因为它写到了对端根本不会读的地方。
- 与其他 composite 一样，自清理尾声会把这两份信用额度再减回去，因此该信号在下次调用时仍可复用。

**成本卡（每 rank）：** 你收到 `(P-1)/P · N` 字节，最终得到 `N/P` 个归约元素——两相 all-reduce（步骤 09）的第一半，其第二半（allgather）你已在步骤 13 构建。

## 边界情况

> **致命陷阱——从错误位置归约自己的分块。** 累加必须从*你的* window 行开始（其中包含你自己的贡献），然后加上每个对端的行。如果你只归约 remote 行，就会缺少自己的贡献，golden 会按已知量失败。**修复：** 在对端循环之前用 `pl.load(data, [my_rank, 0], ...)` 初始化 `acc`。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| golden 差自己的分块 | 累加器从零而非自己的行开始 | 对端循环前加载 `[my_rank, 0]` |
| 每个 rank 得到相同分块 | 归约行固定为 `[0, 0]` | 归约行 `my_rank` |
| 分块边界错误 | 分块偏移计算错误 | `x` 中分块 `c` 在 `[0, c*SIZE]`，`data` 中行 `c` |
| 结果与 torch 不同（容差内） | 每 rank 归约顺序不同 | 用容差比较，而非精确相等 |
| `builtin.tensor.reduce_scatter op must be ReduceOp.Sum` | 用 `op=Max`/`Min`/`Prod` 调用了 **HOST 内置**路径 | HOST 路径上用 `pld.ReduceOp.Sum`，或改用 InCore 复合路径（本步骤的示例），它支持完整家族 |

## 另请参阅

- [05-tutorials](05-tutorials.md) — 教程索引（本步 = 第 14 行）
- [01-collectives](../distributed/01-collectives.md) §ReduceScatter — 完整 API
- [14-allreduce_two_phase](14-allreduce_two_phase.md) — 本步是其第一半的两相 all-reduce
- [04-debugging](04-debugging.md) — 分布式程序的规范故障目录
- 下一步：[20-all_to_all](20-all_to_all.md) — 给每个对端一个不同的 slice
