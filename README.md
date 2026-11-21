# 导频校速 / 补底噪 API（磁带数字化处理）

对数字化后的磁带做两类处理：

1. **带速纠正**：右轨是额定 1 kHz 的正弦导频（瞬时频率保证在 980～1020 Hz，
   即走带速度误差在 ±2% 以内），左轨是节目。服务从导频的**累计相位**恢复整段
   录音时间轴，把节目重采样到均匀时间网格上，输出校正后的单声道 WAV。
2. **补底噪**：降噪门把口述史录音停顿处压成全静音后，按声明的空档样本区间，
   用紧邻空档两侧的原声估计现场底噪并合成补片，把空档填回自然的房间噪声。

## 运行

```bash
docker compose up --build        # 服务监听 http://localhost:8000
```

本地开发（Python 3.11+）：

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

## API

### `POST /correct`

`multipart/form-data`：

| 字段 | 说明 |
| --- | --- |
| `file` | 必填。16 kHz、双声道、16 位 PCM WAV，最长 12 秒。左声道 = 节目，右声道 = 导频 |
| `gaps` | 可选。JSON 字符串，导频缺口的样本区间列表 `[[start, end], ...]`，end 不含，按样本计 |

**成功（200）**：`audio/wav`，校正后的单声道 16 kHz 16 位 PCM 节目。
响应头附带 `X-Output-Samples`、`X-Mean-Speed`、`X-Gaps-Bridged`。

**缺口无法唯一桥接（422）**：不输出 WAV，返回需重录的区间：

```json
{
  "status": "rerecord_required",
  "detail": "pilot gap(s) cannot be bridged unambiguously; re-record the listed intervals",
  "rerecord_intervals": [
    {"start_sample": 24000, "end_sample": 24800, "start_sec": 1.5, "end_sec": 1.55}
  ]
}
```

**输入不合法（400）**：采样率/声道数/位深/时长不符、缺口区间越界或重叠、
未声明的导频脱落、导频缺失或频率超出预期等，`detail` 说明原因。

### `POST /restore-noise`

补回被降噪门压成全静音的现场底噪。

`multipart/form-data`：

| 字段 | 说明 |
| --- | --- |
| `file` | 必填。16 kHz、**单声道**、16 位 PCM WAV，最长 13 秒，即 `/correct` 的输出格式 |
| `gaps` | 必填。JSON 字符串，静音空档的样本区间列表 `[[start, end], ...]`（end 不含）。1～4 段、互不重叠，每段 200～1000 毫秒（3200～16000 样本） |

**参考与合成规则**：

- 空档每侧取紧邻的最多 160 ms（2560 样本）原声估计底噪；遇到另一空档或文件
  边缘即截断。
- 合成补片由 100～1000 Hz、1000～4000 Hz 两个频带的带限高斯噪声组成，频带电平
  从左参考特征经 smoothstep 平滑过渡到右参考，过渡全部发生在空档内部；补片头尾
  各有 5 ms 升余弦接缝淡化，首尾样本精确为 0，避免接缝咔哒声。
- 补片头、尾各 80 ms 在两个频带的 RMS 与**同侧参考**相差不超过 3 dB（实现带
  定点迭代修正，实测误差 ≤ 0.05 dB）。

**成功（200）**：`audio/wav`，16 kHz 单声道 16 位 PCM；总样本数与输入一致，
空档外样本逐样本保持原样（测试以 `array_equal` 校验）。响应头附带
`X-Output-Samples`、`X-Gaps-Filled`。

**参考不足（422）**：本次**不输出 WAV**，返回全部不足的空档及侧别：

```json
{
  "status": "reference_insufficient",
  "detail": "one or more gap sides lack 80 ms of usable original samples or fall below -60 dBFS; no WAV is produced for this request",
  "insufficient_references": [
    {"gap_index": 0, "start_sample": 0, "end_sample": 3200,
     "start_sec": 0.0, "end_sec": 0.2, "side": "left",
     "available_samples": 0, "reason": "too_short",
     "bands_below_minus60_dbfs": []},
    {"gap_index": 0, "start_sample": 16000, "end_sample": 19200,
     "start_sec": 1.0, "end_sec": 1.2, "side": "right",
     "available_samples": 2560, "reason": "band_below_floor",
     "bands_below_minus60_dbfs": [
       {"band_hz": [1000, 4000], "rms_dbfs": -101.82}]}
  ]
}
```

触发条件（任一即列出对应空档与侧别）：该侧可用原声不足 80 ms（1280 样本，
含被相邻空档截断的情形），或该侧任一参考频带 RMS 低于 −60 dBFS。

**输入不合法（400）**：采样率/声道数/位深/时长不符、空档数量不在 1～4、
长度越界、区间重叠或越界、JSON 非法等。

### 演示

```bash
cd demo
python3 make_demo_wav.py        # 重新生成校速演示带
python3 make_noise_demo_wav.py  # 重新生成补底噪演示带（仓库已附带 wav）
bash demo.sh                    # 需要服务已在运行
```

等价的 curl：

```bash
# 8 ms 导频缺口，可唯一桥接 -> 200，得到 corrected.wav
curl -o corrected.wav \
  -F "file=@demo/demo_tape.wav" \
  -F "gaps=[[24000,24128]]" \
  http://localhost:8000/correct

# 50 ms 导频缺口，周期数不唯一 -> 422，返回需重录区间
curl -F "file=@demo/demo_tape_longgap.wav" \
  -F "gaps=[[24000,24800]]" \
  http://localhost:8000/correct

# 补底噪：3 段被门限压静音的空档（200/300/1000 ms）-> 200，得到 restored.wav
curl -o restored.wav \
  -F "file=@demo/gated_silence.wav" \
  -F "gaps=[[16000,19200],[40000,44800],[64000,80000]]" \
  http://localhost:8000/restore-noise

# 补底噪：空档从样本 0 开始，左侧无参考 -> 422，列出参考不足的空档与侧别
curl -F "file=@demo/gated_silence.wav" \
  -F "gaps=[[0,3200]]" \
  http://localhost:8000/restore-noise
```

## 实现说明

核心在 `app/dsp.py`：

1. **导频提纯**：4 阶 Butterworth 带通（900–1100 Hz，`sosfiltfilt` 零相位）。
2. **累计相位 → 时间轴**：对导频做 Hilbert 变换取瞬时相位，`np.unwrap` 得累计
   相位 φ(t)。导频在“磁带时间”上严格为 1000 Hz，因此样本的录音时刻
   τ(t) = φ(t) / (2π·1000)。节目按 τ 非均匀网格插值（线性）到均匀 16 kHz 网格，
   即完成校速。
3. **缺口桥接**：缺口内相位不可观测。对缺口两侧各取一段离边缘 6 ms（避开滤波器
   瞬态）的 12 ms 窗口做线性相位拟合，外推到缺口边缘，得到两端相位；相位差模 2π
   得 δ。设缺口跨过的整周期数为 n，则真实相位推进 2πn+δ 必须落在
   [2π·980·T, 2π·1020·T]（T 为两端锚点间隔）。**当且仅当该区间包含唯一整数 n
   时**补算并线性插值跨过缺口；否则（区间含 0 个或 ≥2 个整数，或缺口旁干净导频
   不足 ~18 ms）整个请求以 422 拒绝并列出需重录的缺口。
   - 推论：缺口可桥接的充要宽度约为 40·T < 1 个周期（T < 25 ms），实际余量还
     取决于真实速度相对 980/1020 边界的距离与相位测量噪声。
4. **自检**：导频缺失（RMS）、未声明的导频脱落（包络检测，提示补填 gaps）、
   实测导频中位频率/分位数超出合理范围，都会以 400 拒绝，避免输出垃圾。
5. 文件首尾的缺口（录音头尾导频缺失）直接裁掉，不参与桥接。

补底噪核心在 `app/noisefloor.py`：

1. **参考窗口**：每个空档两侧各取紧邻边缘、至多 160 ms 的原声窗口；左窗口起点
   不早于上一个空档的结尾，右窗口终点不晚于下一个空档的起点，文件边缘同样截断。
2. **参考充分性判定**：任一侧可用样本 < 80 ms，或 100–1000 Hz / 1000–4000 Hz
   任一频带的周期图 RMS 低于 −60 dBFS，就把该（空档，侧别）记入失败列表；列表
   非空则整个请求返回 422、不出 WAV。两侧检查在合成前一次性完成。
3. **补片合成**：以空档位置派生确定性随机种子，生成白高斯噪声并经两个 4 阶
   Butterworth 带通（`sosfiltfilt` 零相位）切成两频带、各自归一化到单位 RMS；
   频带增益包络在首 80 ms 保持左参考电平、末 80 ms 保持右参考电平，中间按
   smoothstep（3u²−2u³）过渡，全部过渡落在空档内部；两频带相加后乘首尾各 5 ms
   升余弦淡化，接缝样本为 0。
4. **3 dB 验收的定点修正**：淡化和频带叠加会改变边缘窗口的实测 RMS，因此对两个
   平台电平做最多 4 次“实测边缘 RMS → 按比例修正平台增益”的迭代，使补片头尾
   80 ms 的两频带 RMS 收敛到同侧参考（实测偏差 ≤ 0.05 dB，合同要求 ≤ 3 dB）。
5. **样本契约**：输出长度与输入完全一致，仅替换空档内样本，空档外逐样本不动。

## 验证

`python3 -m pytest tests/ -v` 共 **33 项全绿**（原 18 项校速测试无回归，
新增 15 项补底噪测试在 `tests/test_restore_noise.py`）：

- **校速精度**：1.5% 快带、1.5% 慢带、±1.5% 正弦抖晃三种合成带，校正后节目音
  频率回到 440 Hz ±1 Hz，时长伸缩比例正确，`X-Mean-Speed` 与真值吻合。
- **缺口桥接**：10 ms 缺口（含 1.008× 带速误差 + 导频噪声的情形）桥接后与真值
  互相关对齐，残余滞后 ≤3 个样本（少算 1 个整周期会产生 16 样本的偏移，可分辨）。
- **缺口拒绝**：50 ms 缺口（模糊宽度 2 个周期）返回 422 与正确的重录区间；混合
  缺口只报告不可桥接的那一个；首尾缺口正确裁剪。
- **补底噪补片**：单档/双档/四档（含 200 ms 下限与 1000 ms 上限宽度）均 200；
  用独立于实现的 `scipy.signal.periodogram` 代码路径复核，补片头尾 80 ms 两频带
  RMS 与同侧参考偏差 < 3 dB（实测约 0.0 dB）；输出总样本数不变、空档外样本
  `array_equal` 完全一致；同输入两次响应字节相同（确定性合成）；左右底噪电平
  不同的场景下补片电平从左向右平滑过渡。
- **补底噪参考不足**：空档贴文件头/尾（一侧 0 样本）、两空档仅隔 50 ms（内侧
  截断到 800 样本）、纯 500 Hz 正弦（1000–4000 Hz 频带 −101.8 dBFS）均返回 422，
  JSON 列出每个不足的空档索引、侧别、可用样本数与低于 −60 dBFS 的频带，且无 WAV。
- **参数校验**：单声道/双声道、错误采样率、超 13 秒、float WAV、空档数量越界、
  长度越界、重叠/越界区间、非法 JSON、缺失 gaps 字段，均按 400/422 拒绝。
- 演示脚本 `demo/demo.sh` 已对运行中的服务实测全部 5 个请求（2 个 /correct、
  2 个 /restore-noise + health），200 路径输出符合格式的 WAV，422 路径返回预期 JSON。

## 已知限制 / 未完成

- 重采样为线性插值（对 16 kHz 档案语音足够）；如需更高保真可换 sinc/样条插值。
- 可桥接缺口上限约 25 ms（由 980–1020 Hz 约束决定），且缺口两侧各需 ≥18 ms
  干净导频用于相位拟合，否则按不可桥接处理。
- 文件首尾约 6 ms 受滤波器瞬态影响，时间轴有轻微误差。
- 未声明的极短脱落（低于包络检测阈值，约 <1% 时长）不会报错，可能引入局部误差。
- 假设导频幅度基本恒定，未做 AGC；大幅调幅可能影响相位估计。
- 补底噪只按两个频带的 RMS 匹配频谱特征，不重建参考中的窄带信号（如嗡声具体
  频率）；随机种子由空档位置确定性派生，同一输入输出可复现，但补片是统计意义上
  的房间噪声而非原始噪声本身。
- 本环境无 Docker 守护进程，镜像未实际构建；Dockerfile/compose 为常规写法仅静态
  核对，应用本身已在本地以相同入口（`uvicorn app.main:app`）实测。
