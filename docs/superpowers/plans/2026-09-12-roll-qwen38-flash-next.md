# ROLL Qwen3.8-Flash-Next Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a capability-gated Qwen3.8-Flash-Next training path to ROLL with exact GR/PLE/QSA semantics, bounded N-gram memory handling, and reproducible SFT/RL validation on `.181`.

**Architecture:** Extend the vendored mcore adapter with a dedicated `qwen4_exp` model, transformer-native GR/PLE modules, and loss-aware QSA indexer. Keep N-gram storage outside optimizer state by default and expose explicit trainable-table mode only after sparse-row semantics are implemented. Add ROLL lifecycle hooks for capability reports, CPU optimizer forwarding, and versioned weight updates; use vLLM's model mapper exactly once.

**Tech Stack:** Python 3.10+, PyTorch, Megatron-Core 0.16.x, Transformer Engine, flash-linear-attention/FlashQLA, ROLL Ray pipelines, vLLM qwen3_8_flash_next adapter.

**Spec:** `docs/superpowers/specs/2026-09-12-roll-qwen38-flash-next-design.md`

## Global Constraints

- Preserve unrelated dirty files in the parent checkout and keep PR #497 separate.
- Use `/data_hdd/Qwen3.8-Flash-Next` on `.181` as the authoritative BF16 source.
- Default first delivery is architecture-complete LoRA and main-text full-parameter training with the 51B N-gram table frozen; label trainable scope explicitly.
- Reject unsupported long-context sparse training, CP, MTP/vision training, and FP8 rollout until their stated checks pass.
- Production code requires a failing test before implementation; every `.181` claim must include a command and artifact.

---

### Task 1: Establish adapter package and failing contract tests

**Files:**
- Create: `mcore_adapter/src/mcore_adapter/models/qwen4_exp/{__init__.py,config_qwen4_exp.py,modeling_qwen4_exp.py,template_qwen4_exp.py}`
- Create: `mcore_adapter/tests/models/test_qwen4_exp_contract.py`
- Modify: adapter model/config registries and package exports.

**Interfaces:**
- `Qwen4ExpConfig` exposes `hc_count`, `hc_lowrank`, `layer_types`, QSA and PLE metadata without silently dropping unknown checkpoint fields.
- `Qwen4ExpModel` builds a 48-layer hybrid spec and rejects a missing or mismatched pattern.
- Template conversion reports every checkpoint key as trainable, frozen-external, preserved-auxiliary, or derived.

- [ ] Write tests for config registration, 36/12 layer pattern, PP-local layer numbering, and full 1,658-key coverage.
- [ ] Run the focused tests and confirm they fail because registration/module paths are absent.
- [ ] Implement the minimum config, model, template, and registry wiring using the existing qwen3_next/qwen3_5_moe patterns.
- [ ] Run focused tests and inspect the coverage report for zero unknown keys.
- [ ] Commit the isolated adapter change.

### Task 2: Implement exact GR residual path

**Files:**
- Create: `mcore_adapter/src/mcore_adapter/models/qwen4_exp/hyperconnection.py`
- Create: `mcore_adapter/src/mcore_adapter/models/qwen4_exp/hyperconnection_layer.py`
- Create: `mcore_adapter/tests/models/test_qwen4_exp_hyperconnection.py`

**Interfaces:**
- `HyperConnection.mix(stream) -> (stream, block_input, injection_logits)`.
- `HyperConnection.combine(stream, block_output, injection_logits) -> stream`.
- `HyperConnectionTransformerLayer` consumes and returns `[sequence,batch,4*hidden]` and never delegates ordinary residual BDA.

- [ ] Write a zero-block identity test and an FP32 reference test for read/write gates and gradients.
- [ ] Run tests and observe the old implementation's duplicate residual failure reproduced by the fixture.
- [ ] Implement grouped Gemma RMSNorm, SiLU(down / hc_count), sigmoid read mean, and `2*sigmoid(inject/4)` write.
- [ ] Run focused tests, including attention/MLP/hyperconnection input gradient checks.
- [ ] Commit the GR implementation.

### Task 3: Implement exact PLE and frozen N-gram backend

**Files:**
- Create: `mcore_adapter/src/mcore_adapter/models/qwen4_exp/ngram_embedding.py`
- Create: `mcore_adapter/src/mcore_adapter/models/qwen4_exp/ple_layer.py`
- Create: `mcore_adapter/tests/models/test_qwen4_exp_ple.py`

**Interfaces:**
- `FrozenNGramEmbedding.forward(input_ids) -> [batch,seq,ple_embed_dim]`, with checkpoint-provided hash constants and EOS segmentation.
- `PLELayer.forward(stream, input_ids) -> stream_delta` with key width `4*hidden`, value width `hidden`, and dilation `3`.

- [ ] Write tests for hash IDs, EOS isolation, checkpoint-shaped parameters, and residual gradient flow.
- [ ] Run them against the old adapter and confirm shape/semantic failures.
- [ ] Implement a non-persistent frozen table with bounded CPU/pinned staging hooks; add a separate explicit row-sharded trainable interface without enabling it by default.
- [ ] Run focused tests and verify table is absent from named parameters, state dict, and optimizer groups.
- [ ] Commit the PLE/N-gram implementation.

### Task 4: Implement QSA indexer loss and sparse training guard

**Files:**
- Create: `mcore_adapter/src/mcore_adapter/models/qwen4_exp/qsa.py`
- Create: `mcore_adapter/tests/models/test_qwen4_exp_qsa.py`
- Modify: model forward/loss interface and ROLL training config validation.

**Interfaces:**
- `QSAIndexer.forward(hidden, position, visible_mask) -> scores, selected_indices`.
- `qsa_indexer_kl_loss(scores, teacher_attention, complete_blocks, selected_blocks) -> scalar`.
- `Qwen4ExpModel` rejects sequences >2048 when sparse training kernel is unavailable.

- [ ] Write boundary tests for lengths 2047/2048/2049/2052/4096, padding, EOS, tie-breaking, and KL stop-gradient.
- [ ] Run tests and confirm no dense fallback is accepted above the configured budget.
- [ ] Implement FP32 reference selection and an autograd-safe path; keep the full `[B,S,S]` mask out of the training implementation.
- [ ] Run focused tests and compare selected sets and gradients with the official Transformers reference.
- [ ] Commit QSA support.

### Task 5: Fix conversion, optimizer, and ROLL/vLLM lifecycle

**Files:**
- Modify: mcore adapter converter and `roll/third_party/megatron/model_update.py`.
- Modify: `roll/distributed/strategy/megatron_strategy.py` and offload patch.
- Modify: `roll/third_party/vllm/worker.py` and compatibility layer.
- Create: focused conversion/lifecycle regression tests.

**Interfaces:**
- Conversion uses stacked GDN/SwiGLU components, TP=2/EP=8 constraints, cloned expert slices, and complete coverage reports.
- `OptimizerConfig` receives CPU-offload fields and distinguishes true CPU optimizer from phase state offload.
- Weight updates carry version/layout metadata, apply model-owned mapper once, invalidate QSA/GDN/PLE caches, and acknowledge all ranks.

- [ ] Write failing tests for optimizer field forwarding, duplicate mapper application, unknown keys, and duplicate residual lifecycle.
- [ ] Implement minimal fixes and run focused tests.
- [ ] Run TP1/TP2 conversion and roundtrip checks on small fixtures; verify source tensor slices bytewise.
- [ ] Commit lifecycle fixes.

### Task 6: Build `.181` validation runner and execute full acceptance matrix

**Files:**
- Create: `scripts/roll_qwen38_flash_next_validate.sh` and Python probes under `output/roll-qwen38-flash-next-research/validation/`.
- Create: `output/roll-qwen38-flash-next-research/validation-report.md`.
- Create: `.181` configs for LoRA SFT/RL and frozen-table text training.

- [ ] Run V0-V3 component/distributed checks on idle `.181` GPUs.
- [ ] Run V4 48-layer real-weight 2K/8K forward/logprob parity and memory sampling.
- [ ] Run V5 at least 100 optimizer steps for SFT in both declared modes, with save/resume.
- [ ] Run V6 at least 20 RL rollout/update/reload cycles and verify versions, advantages, and reference stability.
- [ ] Run V8 20-step throughput repeats for baseline and optimized kernels.
- [ ] Record all failures and unsupported combinations; do not label incomplete modes as supported.
- [ ] Commit validation runner and report.

### Task 7: Review, upstream PR, and final handoff

2026-09-17 用户授权范围补充，提交前还需完成：

- [x] 复现并修复 RLVR/Agentic 在 LoRA OPD 时跳过显式教师的问题；`.181` CPU 从 4 failed/10 passed 到教师选择与 RL RNG 恢复合计 17 passed。
- [ ] 为 Flash-Next 主干全参数与 LoRA 准备纯 OPD 验收，使用真实不同权重的冻结教师与学生 rollout，验证教师概率确实决定逐 token KL/优势。
- [ ] 在 `.181` 执行 OPD 更新、权重同步、保存和恢复后新更新，检查教师权重不变；分别记录容量、数值与任务质量，不能合并宣称通过。

- [ ] Run full relevant local tests and verification-before-completion checks.
- [ ] Review code diff and generated coverage/memory reports.
- [ ] Split generic ROLL fixes and Qwen3.8 model support into reviewable commits/PRs.
- [ ] Push branch and create official ROLL PR only after `.181` evidence meets the acceptance matrix.


2026-09-13 08:30 UTC: full coldresume v6 session56739 exited1 at08:12:54 with GetTimeoutError during worker checkpoint upload (default rpc_timeout3600s). All8 restore+step2 succeeded; loss2.2245461866259575/gradnorm62.881524962747406/heldout2.667731821537018/train102.576s. Full32shards1,769,920,831,886B exist in staging, but HDDpartial~615GB lacks pipeline state because BasePipeline saves it after worker completion. No completion.json or fullsave claim. All8 GPUs released. Profiles archived full-v6/summary.json:8load returns, minhost485925281792B, peakrankRSS247981305856B. CandidateSFTconfigs rpc_timeout14400 nowlocal. Preserve validv9 and incompletev6 staging.
Isolated regression session5270exit0:14passed868.16s incl realCUDA main/LoRA. ActualSFTConfig and nativeLoRA artifact metadata session10293exit0. SFTupdate profiler session69007exit0:realRay worker calls actualMegatronTrainStrategy.train_step withtinyQwen4; backbone+LoRA each2successfulupdates, requiredGR/PLE/QSA/indexer/GDN/expert/shared groups changed, noerrorlogs, peakCUDA511468032B. This verifies diagnostics only, not fullmodelLoRA capacity or100steps.
Scopedreview recovery-validator-review2 closes tracebackP2, reports3validatorP2 (tracker, assetcontract, scheduler/RNGstructural). Originalimplementer fixing; tracker12tests passed, remaining expanded RED/GREEN inprogress. Existing approvals sufficient; nocurrent approvalblock. FlashNext numerical/SFT100/RL20/export/fault/performance/vendor/PR gates remain open.


2026-09-13 09:20 UTC: actual LoRA smokev1 session27175 EXIT1. Complete model load +heldout+trainingforward/backward reached optimizerstep; nativeTE multi_tensor_scale in globalgradientclipping exhausted20321handlepool. Zero successful optimizerupdates; perrank431841280trainable/20047192960frozen parameters; peakCUDA45781952000B (~42.64GiB), rank7 diagnosticextra45826157056B. Trace archive full-lora-smoke-v1/summary.json. vLLM API incompatibilities independently reproduced and fixed byroot: RED7failed2passed -> GREEN22passed1skip; same installedCPUtensorprobeGREENexit0. No rolloutclaim, remote fix isolatedonly.
TE capacity diagnostic with24000tinygradients reproduces2fail7.79s. OfficialContext7 documents20MiBdefault andpoolsizeenv, but64MiBsetting remains2fail7.55s and compiledlibrarystrings lacktheoption; no suchunverifiedconfigdeployed. GuardedMegatronclip patch nowbatches2048 gradients usingoneunchangedglobalcoefficient. BeforeSHA4a68e9aae8cb1188f30766af93fabba00a43a8512f8fe3e9e4154d80de4c84ee→after a7a40ac42a1658fad4f589b0da5d2963cadcb5a3c01dfffc7c4b6d8174cf4556; appliedviawritable/data_nvme/workspace/roll-qwen38-validation/Megatron-LM. Default20MiBpool combinedGPUregression4passed39.51s, includingFP32/BF16 fourexactrepeatedclips and actualbackbone/LoRAcheckpointnextupdate.
ActualLoRAsmokev2 launchedsession40234 withfreshRay26402/dashboard28291 andunchangedmodel/data/seed/config. output/sft-lora-pipeline-smoke-v2.log; profiles output/sft-training-metrics/full-lora-smoke-v2. It isrunning, notyetpassed. Broadvalidatorfix28CPUtests plusrealv9/tinyLoRAartifactchecks passed; independentre-reviewbriefpreparedbutfollowuproutingfailedandmisroutedIgnoreonce, usertoldnoanswerneeded. Donotmisuseasyncinputforagentcontrol. Allmajor100SFT/20RL/numerical/export/perf/vendor/PRgatesremainopen.


### 2026-09-13 09:41 UTC: LoRA 实模两步通过，冷恢复与 RL 转换继续

- `.181` LoRA TP1/EP8/ETP1、rank64、CPU Adam 完成两次更新并正常退出（session40234，exit0）。训练 loss 2.414041629 → 2.196113424；heldout loss 2.588216335 → 2.577569250。这是 8 条训练/8 条 heldout 的 smoke，不是长期质量结论。
- `completion.json` 确认 8 份 adapter、16 个优化器文件、8 份 worker RNG、pipeline CUDA RNG、scheduler step2 和冻结资产清单。八 rank 均记录成功更新、有限参数值；GR/PLE/QSA/indexer/GDN/专家/shared expert 的有界采样在两步内均变化。峰值 CUDA 45,970,934,272 B（42.81 GiB）。
- 从该检查点启动实模冷恢复（session13630，driver484235，Ray26403/dashboard28292），沿用同一数据与常量学习率，把总 epoch/step 延长到4，继续两次更新。这不是与无中断轨迹逐张量比较。
- vLLM API 修复独立复审 spec PASS、quality APPROVE；实际安装版本的 CPU tensor loader 精确通过。实模 rollout 未通过，主远端源仍未同步这项修改。
- 实际 MCA CPU 转换复现 GDN/GR adapter 缺分布规则、专家 adapter 缺转换 op、QSA gated LoRA B 输出形状错误。已建立独立转换修复子任务；不能据 SFT 通过宣称 RL 可用。
- 检查点校验器复审确认 tracker 修复，但资产路径/空清单和 RNG 内部结构仍有假阳性，正在按真实 native loader 合同修正。
- 全主干完整恢复后保存、LoRA/主干各100步、RL20轮、2K/8K完整数值门槛、导出迁移故障和性能验收仍未完成。保留 v9 有效主干检查点与 v6 完整 staging；Flash-Next 未同步 vendor、未提交 PR。

证据：`output/sft-training-metrics/full-lora-smoke-v2/{completion,summary}.json`、`output/sft-lora-pipeline-smoke-v2.log`、`output/sft-lora-pipeline-coldresume-v1.log`、`output/dependency-source-vllm/qwen4-lora-conversion-red-v1.log`。


### 2026-09-13 10:07 UTC 更新

LoRA 2K 冷恢复和真实 8K 长文本 smoke 均已正常退出并完成检查点校验。冷恢复继续两步，scheduler 到4；8K 两步 loss 1.382071897→1.339245498，所有 rank 的七类训练参数采样均变化，峰值 CUDA 57,069,223,424 B。证据分别在 `output/sft-training-metrics/full-lora-coldresume-v1/` 与 `full-lora-long-smoke-v1/`。

100 步 8K LoRA SFT 已启动（session33222，driver534002），固定400条训练/256条heldout；已完成首步并继续运行，尚无100步完成结论。完整主干SFT、数值一致性、RL20轮、精确连续/恢复对账、导出/故障/性能门槛仍开放。

校验器新增损坏用例40项通过，root对三个真实检查点复核通过。round3修复由root做范围复审，见 `validator-review4-report.md`；不声称另一子代理复审通过。vLLM API修复已同步主远端，实际本地adapter文件读取探针逐元素一致。RL候选配置及80/16公开GPQA划分通过实际严格配置解析和编码检查；还未启动RL。LoRA转换新回归在真实CPU环境RED9failed1passed，修复进行中。


### 2026-09-13 11:37 UTC continuation

100-step 8K LoRA SFT is still running, with95 completed updates at11:36:07UTC. Both long-input phases contain actual8192-token batches. checkpoint50 is retained and passes the latest native structural validator:8adapter payloads,16optimizer files,8worker RNGs,pipeline CUDA RNG,scheduler51. Heldout loss at step0 was1.47875770 and before step50 was0.81504061; final heldout not measured yet. No100-step completion claim.

Final LoRA template/streaming CPU suite20passed2skipped11.32s. Public adapter export fix adds Qwen4-only direct vLLM2D serialization (not generic HF PEFT reload): RED6failed13.22s -> GREEN26passed2skipped13.46s. Complete real adapter export passed in25.03s, file3,515,704,472B/148808tensors. Native vLLM file loader/EP filtering and CPU packing passed for all48layers on EP0/EP7, each9892projections/276packed modules, actual ranks6/64 with exact scale2, zero CUDA allocation. Full model adapter forward/RL remains unverified. Independent artifact review identified2P1s now fixed and root-reviewed; independent fix re-review remains pending because collaboration control is unavailable. Detailed report: lora-artifact-fix-report.md.

Next GPU job is the prepared identical100-step configuration resumed from checkpoint50 to99, with final heldout evaluation after saving. Existing run must exit and release GPUs first. A full adapter/Adam/RNG checkpoint comparator passed a native small-checkpoint positive/negative check: identical accepted, separate LoRA/Adam changes detected, missing RNG rejected. Evidence output/sft-training-metrics/comparator-selfcheck-v1.log. Main remote production code is not changed by the isolated artifact work; no vendor sync or Flash-Next PR. All full-backbone, exact restore, numerical, RL, export lifecycle and performance gates remain open.


### 2026-09-13 12:08 UTC: 100-step LoRA complete; exact cold-resume comparison running

The continuous real-model LoRA SFT run exited 0 and all eight ranks completed 100 successful updates. The final checkpoint99 passes the latest native structural validator (scheduler100, eight adapters, 16 optimizer files, eight worker RNG files, pipeline CUDA RNG and frozen-asset manifest). Peak CUDA allocation was 57,704,576,512 B (53.74 GiB/card); minimum host available was 1,584,964,308,992 B. All seven trainable groups had finite changed bounded samples. This is mixed effective sequence length under an 8K window: 20 updates had actual8192-token batches; it is not 100 fully populated8K updates or a full frozen-parameter checksum. Heldout loss was1.47875770 atstep0 and0.81504061 beforestep50; the original run has no final heldout measurement. Evidence: output/sft-training-metrics/full-lora-100step-8k-v1/summary.json, checkpoint99-validator-v1.log, and output/sft-lora-pipeline-100step-8k-v1/completion.json.

The identical configuration resumed from retained checkpoint50, keeping total100steps, seed42, constant LR and input hashes. Active driver560995/session54613, Ray26406/dashboard28295; output/sft-lora-pipeline-100step-8k-resume-v1.log. Fresh SSH at12:05UTC confirmed the driver and all8 active GPUs. Newly executed steps51–66 have exactly matching loss, global gradient norm and batch mean length against the continuous trajectory. Earlier rows loaded from checkpoint history are excluded from this evidence. Final full adapter/Adam/scheduler/RNG/pipeline comparison waits for checkpoint99. The output-only launcher adds a native final heldout evaluation after the final save. Do not modify the running source/config, launch anotherGPU job, or stopRay globally.

Artifact serializer fixes remain isolated pending scoped independent review; complete real-file native vLLM CPU load/packing evidence covers all48layers and EP0/EP7 but not GPU forward. Main-backbone100steps and complete re-save, full numerical gates, RL20cycles, deployment/export/fault/performance gates, vendor sync and Flash-Next PR remain open. Existing user authorization remains sufficient; current SSH access works with the required sandbox network escalation.


### 2026-09-13 12:26 UTC: artifact review closed and CPU attribution recorded

Independent artifact scoped re-review is now spec PASS / quality APPROVE with nonblocking follow-ups (`lora-artifact-fix-rereview-v1.md`). Root added public-convert negative coverage for nonuniform rank/alpha, bias/DoRA/rsLoRA/modules-to-save, conflicting duplicate tensors, empty stream and malformed names/dimensions, including no partial output when a later named adapter is invalid. Exact current post_converter source plus tests uploaded to isolated regression directory only; combined CPU artifact/conversion/streaming suite36passed2skipped13.89s, session94493exit0, log output/dependency-source-vllm/lora-artifact-coverage-v2.log includes source hashes. Multi-adapter export still retains all adapters at once (~3.5GB perrealadapter); this limitation remains explicit.

CPU-only real-input GR attribution used exact pinned HF classes and current MCA hyperconnection source with CUDA uninitialized, four threads. For code2K/layer17 and en/zh8K/layer12, grouped sum/div normalization equals native mean exactly; FP32 block relativeL2 is3.33e-8–3.73e-8. Native BF16 gate accumulation differs from MCA FP32 gate accumulation by~.0027; this does not explain the observed full-FP32 cumulative gate failure. No production arithmetic or tolerance changed. Evidence output/dependency-source-vllm/gr-cpu-attribution-v1.{json,log}, sourcehashes included. Prepared output/numerical-layer-replay/hf_routing_attribution.py for selected actual GPU layer replays after the recoveryjob releases GPUs. Forced selections are diagnosis only, never natural parity acceptance.

Recovery actual steps51–84 loss/gradnorm/batch lengths match continuous records exactly; final checkpoint tensors and finalheldout remain pending. Fresh .181 storage:283GiB NVMe free,12TiB HDDfree,~1.45TiB MemAvailable. Main training sources/config unchanged. NativeGPU adapter forward, fullRL and remaining release gates remain unverified.


### SFT pipeline RNG recovery contract (2026-09-13 continuation)

The saved pipeline RNG must be restored before new driver-side data work. Resume skips completed epochs without constructing their DataLoader iterators. For a checkpoint inside an epoch, reconstruct the current iterator and discard consumed batches, then restore the checkpoint RNG before fetching the first new batch; at an epoch boundary the new iterator consumes its seed normally. This preserves Python/NumPy/Torch driver RNG and single-process randomized transforms through subsequent validation and epoch transitions. Missing RNG must fail before new training updates. Multiprocess DataLoader worker-local stochastic state is not captured by the existing checkpoint format, so exact stochastic multi-worker data replay is not claimed. The approved fixed public validation uses num_workers=0.

Actual SFT run/DataLoader/native RNG serialization regression (three checkpoint positions: middle of first epoch, epoch boundary, middle of second epoch) currently fails3cases11.77s before implementation: output/sft-training-metrics/pipeline-rng-red-v1.log. The existing100step recovery uses the unmodified pipeline; its fullcheckpoint comparison remains separate evidence.


### 2026-09-13 12:55 UTC: 100-step recovery complete; pipeline RNG defect isolated

Recovery session54613 exited0 with49newupdates(steps51–99), completecheckpoint99 and finalnativeheldout.9482877074660792(256examples). Initialheldout1.47875770 andstep50heldout.81504061; finalimprovesinitialbutregressesfromstep50. All8rank traces complete andfinite, sevenrequiredtrainablegroupschanged; peakCUDA57,700,410,880B, minhostavailable1,591,381,143,552B. Continuous/recovered49steps match loss,gradnorm andinputlengths exactly. Latestnativevalidator session78685exit0 verifies fullcheckpointstructure/scheduler100/assets/RNG.

Fulltensorcomparison v2 covered408,806tensorleaves/7,475,287,104elements and297,936DCPstorageitems. AllLoRA,Adam,scheduler,eightworkerRNGsandotherpipelineRNGstateexact. Only mismatch: pipeline/rng_state_pipeline.pth/cpu (twoentries,maxabs4). v1comparison failedclosed onincorrectrootdist_optimizerdirectory; v2usesactualiter_*/dist_optimizer andpassesupdatedactualDCP positive/negative selfcheck before realcomparison. Evidence output/sft-training-metrics/resume-checkpoint-exact-v2.{json,log}, full-lora-100step-8k-resume-v1/summary.json and resume-checkpoint-validator-v1.log. This is not fullRNG-exactacceptance.

Root repaired SFTPipeline driverRNGrecovery accordingtothe appendedcontract. ThreeactualDataLoader/randomtransformcontinuationcasesRED3failed11.77s -> combinedGREEN8passed12.10s,includingmissingRNGrefusalandexistingworkerAdamresume. Source/tests onlyisolated; no mainremoteproductiondeploymentorfullmodelrerunyet. Scopedreviewbrief sft-pipeline-rng-review-brief.md exists, but collaborationrouteagainmalformed; noindependentreviewclaim. maxsteps800 earlier suspiciondisproved: actualstrategy dividesbyDP8and logsworker100before schedulerconstruction; no schedulerfixneeded.

NativevLLMGPUprobe v1 exited1beforeengineconstruction duea diagnosticassertionconfusing publicre-exportpathwithclassdefinition. Correctinstalleddefinition isvllm.models.qwen3_8_flash_next.nvidia.model.Qwen3_8FlashNextForConditionalGeneration; fixedprobev2nowrunning(session60112),EngineCore586078,TP8/EP,BF16,eager,rank64LoRA. Log output/dependency-source-vllm/native-lora-gpu-probe-v2.log;remoteJSON output/dependency-source-vllm/native-lora-gpu-probe-results-v2.json. Root addedexplicitcustom_init_workerRPCafter LLMconstruction (nativeLLMdoesnotcallROLLextensionhookautomatically), plus launcher ownenvironmentandidleGPUcheck. Twoearlierlongcommandapprovalstimeoutbeforeexecution; simplifiedreviewablescriptcommandexecutednormally, nocurrentpermissionblock. Do notlaunchotherGPUtasksuntilthisengineexitsandGPUsrelease.

HF routingattributionprobe ispreparedanduploaded output/numerical-layer-replay/hf_routing_attribution.py; stillnotexecuted. Allfullnumerical/RL20/backbone100+re-save/fault/performance/vendor/FlashNextPRgatesremainopen.


### 2026-09-13 13:16 UTC: native LoRA lifecycle passed; routing attribution completed

The real native vLLM TP8/EP LoRA probe v3 exited 0. All ten assertions passed: baseline replay with adapter disabled, real adapter effect (56/68 prompt-token logprobs changed), eight-rank load/removal acknowledgements, exact adapter reload, and exact base/adapter replay after sleep1/wake. Three Chinese/English/code prompts used 23/22/26 input tokens. Evidence: `output/dependency-source-vllm/native-lora-gpu-probe-results-v3.json` and the v3 log. This is a small native inference lifecycle probe, not full ROLL RL, cross-framework parity, 20 weight versions, or performance acceptance. Shutdown logged forced EngineCore termination and a shared-memory warning; fresh SSH at 13:16 UTC confirmed all eight GPUs at 0 MiB.

HF single-layer routing attribution v1 exited 0 and exactly reproduced all three saved HF outputs with FP32 and TF32 disabled. Code2K layer15 changed one token's top-10 expert set across the two input trajectories; English8K layer8 changed two. Aligning selected sets reduced trajectory errors from 2.91e-4 to 1.67e-6 and 3.88e-4 to 9.87e-6, respectively. Chinese8K layer6 changed no expert set between the two HF trajectories, but same-input HF/MCA error remains 7.11e-4 and needs another attribution step. Forced selection is diagnostic only; the complete natural numerical gate remains failed. Records: `output/numerical-layer-replay/hf-routing-attribution-v1/{summary.json,events.jsonl}` and the v1 log.

SFT driver RNG fix remains covered by the earlier 8-pass CPU regression; independent scoped review is now dispatched through the actual agent tool. Corrected full-model resume-v2 launcher is prepared with separate output and Ray ports, preserving continuous checkpoint50/99. It has not started. The latest validated converter/template and pipeline RNG production changes are still isolated, pending deployment before the relevant acceptance run.

Ruling: retain all original continuous and resumed checkpoints as immutable comparison evidence. Close pipeline RNG equality with a fresh identical continuation, and continue routing diagnosis without relaxing numerical tolerances. Full-backbone 100 steps/re-save, full RL, export/fault/performance gates, vendor synchronization and Flash-Next PR remain open.


### 2026-09-13 13:53 UTC: reviewed fixes deployed; real RNG recovery rerun active

The independent SFT driver RNG scoped review is spec PASS / quality APPROVE (`sft-pipeline-rng-review-report.md`). Root deployed the reviewed pipeline plus reviewed Qwen4 template/post-converter to the main validation container after confirming all eight GPUs idle. All three source SHA256 values exactly matched the local `output/sft-training-metrics/acceptance-deploy-v2-manifest.json`. Previous production files are retained in container `/tmp/roll-qwen38-pre-acceptance-v2.tar`. Existing checkpoints are untouched. This deployment also makes the reviewed adapter conversion available to the subsequent ROLL RL run.

The corrected full-model recovery v2 is ACTIVE: local session18041, driver590612, Ray26407/dashboard28296, output `output/sft-lora-pipeline-100step-8k-resume-v2`, checkpoint timestamp `20260913-134252`. All eight workers restored checkpoint50; first new step51 began13:48:18UTC. Steps51–53 now match uninterrupted loss, global gradient norm and effective input length exactly. Keep GPUs exclusive to this job and do not mutate its sources/config. Final99/full tensor-RNG equality is pending.

Single-layer diagnostic v2 exited0(session32984). All three HF saved outputs exactly reproduced again. Switching from full-softmax/topk to logits-topk/selected-softmax changed zero expert sets in these cases and did not resolve Chinese8K layer6 same-input error(.00071133). This rules out that proposed explanation in the observed layer; no production routing change is warranted from it. Record: `output/numerical-layer-replay/hf-routing-attribution-v2/summary.json`. Natural numerical acceptance remains failed.

Root prepared `output/numerical-layer-replay/mca_single_layer_capture.py` plus launcher to instantiate the actual original layer6 spec only, load converted real weights under TP2/SP+EP8/ETP1, and capture GDN/MLP/shared/router components on identical saved input. Syntax/help/shell checks passed; the distributed capture is NOT executed and is queued after SFT. It must first reproduce the historical MCA layer output before attributing those old measurements. No full-model parity claim follows from the prepared script.

Operational note: scp approval review timed out twice; SSH transfer to host /tmp succeeded. The target output directory is container-owned, so files were copied into the existing container with docker cp then unpacked there. No current authorization gap. An attempted additional agent dispatch was misrouted as an Ignore prompt; root resumed the diagnostic preparation locally. Do not use user-input tools for agent control.


### 2026-09-13 14:21 UTC: LoRA admission race reproduced and repaired locally

Static independent preflight (`rl-candidate-preflight-v2.md`) found an unawaited colocated `add_lora.remote` and lost rank replies. Root added an unchanged-production-method test chain through InferWorker, VllmStrategy, CustomAsyncLLM and MegatronWeightUpdater, with real asyncio and a pending Future replacing the external engine/Ray transport. RED v1 failed on the test's wrong owner class name and is not bug evidence. Corrected RED v2 reproduced all eight intended failures: early update completion, swallowed engine error, discarded success replies, and accepted missing/failed rank replies. Minimal fix waits for final admission, propagates the native reply list, and rejects cardinality or boolean failures. Combined local suite:30passed1skipped(.21s), `lora-admission-green-v2.log`; v1 green had two obsolete Receiver fixtures missing the rank contract, now corrected. This verifies the source boundary; real Ray/native GPU repeated-adapter admission still needs validation. Files/hashes are listed in `lora-admission-source-manifest-v1.json`. Changes remain LOCAL ONLY, not deployed into the active SFT source.

Fresh .181 data read verifies the candidate GPQA file exists with exactly80records and SHA256 f5332641040e99df916b9318730953c8240ddbe676c178e2dfc294302c249570. The preflight's missing-local-file issue is not a missing remote dataset. Absolute paths will be used for data/output/checkpoints and rollout dumping. Full RL20/version/reference/advantage evidence remains unimplemented/unexecuted.

Additional static phase review found two issues requiring targeted reproduction before RL: MegatronTrainStrategy load_states/offload_states only invoke the frozen-module helper when include is explicit; include=None currently skips it. Also start_model_update reloads actor model parameters before transfer, whereas custom_add_lora wakes the full inference base before that sender context exits. With a ~40GB frozen actor plus ~44GB inference base percard, simultaneous residence may exceed80GB. These are static/source-budget findings, not fresh full-GPU OOM evidence. Investigate with actual tiny/native offload and admission sequencing; do not launch unattended full RL merely because the new Future tests pass. Keep the active SFT untouched.


### 2026-09-13 15:01 UTC: SFT exact recovery closed; LoRA phase regression in progress

The full 8K LoRA continuation v2 exited 0 after all 49 new updates (steps51–99). Freshly archived rank traces confirm all eight ranks updated successfully, all watched trainable samples finite, and GR/PLE/QSA/indexer/GDN/experts/shared groups changed. Loss, global gradient norm and effective input min/max/mean match uninterrupted training at every new step. Final heldout256 loss is0.9482877074660792; initial1.47875770 and intermediate0.81504061, so no monotonic-improvement or downstream-ability claim. Fresh continuation CUDA peak57,700,410,880B and min host available1,586,957,209,600B. Evidence: output/sft-training-metrics/full-lora-100step-8k-resume-v2/summary.json.

Full checkpoint comparison v3 is exact=true, failure_count0 across408,806tensor leaves/7,475,287,104elements,8adapter payloads,297,936optimizer storage items,8worker RNG files and pipeline RNG/scheduler/step. Structural validator also passed. Evidence: output/sft-training-metrics/resume-checkpoint-exact-v3.json and resume-checkpoint-validator-v2.json. Checkpoints compared are continuous20260913-095906/checkpoint-99 and resume20260913-134252/checkpoint-99. This closes exact deterministic resume for this LoRA configuration (num_workers0,shuffleFalse), not arbitrary multiworker loader recovery or the backbone/full RL gates.

Independent LoRA admission review: spec PASS / quality PASS (lora-admission-review-report.md); Linux31passed evidence matches current six-file hashes. Main validation source not yet deployed with admission fix. Legacy vLLM online-LoRA remains outside verified support.

New real tiny Qwen4/LoRA/CPU Adam phase regression v1 failed before target assertions because launcher TORCH_COMPILE_DISABLE=1 conflicted with FlexAttention; this is harness failure, not production evidence. Removing that launcher override produced intended RED2 in26.08s: default offload retains frozen CUDA weights, and real Worker model-update admission sees frozen actor base on CUDA. Minimal isolated repair adds strategy-specific model-update load kwargs and symmetric include_frozen_parameters handling, restores include=None frozen offload/load. Native checkpoint/next-update GREEN is running session94683. Evidence logs: output/dependency-source-vllm/lora-frozen-phase-red-v{1,2}.log and green-v1.log. No full GPU RL capacity claim.

Natural numerical acceptance remains failed. MCA layer6 component capture is queued after phase regression; no tolerance changes. Backbone100steps/full re-save, RL20/version/reference/advantages, export/fault/performance gates, final review/vendor sync/Flash-Next PR remain open. Preserve valid v9 and all v6 staging; avoid another1.77TB NVMe save.


### 2026-09-13 16:29 UTC: exact LoRA phase recovery repaired; real RL smoke prepared

SFT LoRA100-step8K continuationv2 is fully archived and exact:49new updates, loss/gradnorm/input metrics unchanged, finalcheckpoint comparison408,806tensorleaves/7,475,287,104elements failure_count0 includingalladapter/Adam/scheduler/worker+pipelineRNG. Finalheldout256loss.9482877074660792. ContinuationpeakCUDA57,700,410,880B, minhostavailable1,586,957,209,600B. These completedresults belong to the recorded SFTsource; subsequentphasefixes have only small/native anddistributedregressionevidence.

LoRA phase defects repaired in local+isolatedsource: defaultincludeNone nowhandlesfrozenparams; strategy-specific model-updatekwargs keepLoRAfrozenbaseCPUwhileadaptersCUDA; allchainedoptimizerparams restorebeforeDDPhooksregister; frozenpacking uses256bytealignment. Diagnosticchain separates two independentcauses of nextupdate drift: missingexpertgradientsfromprematurehookregistration (optimizer-only/individual-frozenRED2->GREEN2), and residualroundingfromunalignedfrozenpacking (alignedpackingdiagnosticGREEN1). Earlierstatement thatpackingwasruledout is superseded: itwasnot theonlycause, butwasthesecondcause. No tolerancechange.

Finalactualdefault/admissionnativeLoRAphaseGREEN2in35.43s; CPUbufferalignmentGREEN5in3.23s. Broaderregression14passed44.98s (backbone+LoRAcheckpoint andCPUoptimizer), then4rankDP4/EP2hybridlifecycletests3passedoneachrank35.36–35.71s, session4560exit0. Logs output/dependency-source-vllm/{lora-frozen-phase-green-v3,buffer-alignment-cpu-green-v1,offload-distributed-regression-v1}.log.

Actualadapteradmissionfailure thenreproduced actorCUDAretention (RED1,22.83s). state_offload_manger nowexecutesexistingoffloadinfinally andrestoresprior roll_EXEC_FUNC_NAME. All3nativecases(default,admission,admission_failure) GREEN3in44.31s, session40910exit0; exactcheckpoint/gradients/Adam/RNG/nextupdateallretained. Thiscoversoperationfailureafterload, notpartialload/cleanupitself failing orallRLfaultgates. Source/tests: newtests/third_party/megatron/test_lora_frozen_phase_offload.py andtests/utils/test_offload_buffer_alignment.py. Mainremote training sourcehasnotbeendeployedwiththesephasefixes.

IndependentadmissioncompletionreviewPASSED(lora-admission-review-report.md); independentphase/offloadreviewremainsPENDING becausemultipletoolcallsweremisroutedintoIgnoreprompts. Rootmustnotclaim independentapproval. Reviewbrief/package/rootreportprepared. Ruling: continue reversiblevalidationinanewisolatedsource snapshotwhileindependentreviewisunavailable; finalupstreamPRstillrequirescompletegatesandreview.

Chinese8Klayer6captureandHFcomponentattributionbothEXIT0. All8MCA ranksreproducedhistoricaloutput;HFhistoricaloutputandselfrepeat exact. Samewhole-layerinputerror.00071133 amplifies atonechangedexperttoken6135 (HF499vsMCA311,margin1.43e-6). WithsamecapturedMLPinput,routerlogitsexactandallsetsidentical;MLPerror7.19e-7. SamecapturedGDNinputstill1.24e-6. ForcedMCAselectionreduceslayererrorto4.25e-7, diagnosticonly. Naturalfullmodelgate remainsfailed. Evidence output/numerical-layer-replay/hf-mca-components-zh6-v1/summary.json andzh-layer6-attribution-report.md. No numericalproductionchanges.

PreparedactualRL2step smoke: output/rl-validation-data/run_rl_smoke.py, rlvr-lora-smoke-v1.yaml, run_rl_smoke_v1.sh. Uses80GPQArecordswithknownSHA,seed42,absolute model/data/output/checkpointpaths, TP1/EP8actor,TP8/EPinfer,BF16,CPUAdam,enabledboundedlogprob/entropy. Driverrecordsactualcompute_advantageandmodel_update events; existingrealworkerupdateprofilerrecordsboundedparamdeltas. Doesnotclaimfullfingerprints/referenceor20versions; finaltrainedv2isnotreloadedinthissmoke. Complete636file/~8MBsource snapshotisolatedat /tmp/roll-qwen38-rl-smoke-v1/ROLL; manifestSHA7fe888ffcd930b28f00b22849cb87f6320fec7ae86af4409b536ed909cf0c914. Mainmodelcache/outputassetsreferencedabsolutely. CPUtypedconfigpreflightpendingresult; GPUrunnotstartedasyet. Plannedoutput mainROOT/output/rl-lora-pipeline-smoke-v1,Ray26410/dashboard28299.

Remaining: natural2K/8Knumerical, mainbackbone100steps+completecold-resume re-save, fullRL20/version/reference/advantage/reloadvalidation, export/fault/performance, independentfinalreview/vendor sync/FlashNextPR. Preservevalidv9andfailedv6fullstaging; noadditional1.77TBNVMecheckpoint.


### 2026-09-13 17:31 UTC: real RL Ray initialization failure reproduced and repaired

The v1 real ROLL RL smoke (session92417) exited1 before any RL update. The inner traceback is CustomRayDistributedExecutor._init_workers_ray calling a removed native RayWorkerWrapper.get_node_and_gpu_ids RPC. Fresh SSH confirmed driver620540, InferWorker643409 and EngineCore643871 absent and all eight GPUs at0MiB before scoped regression. This supersedes earlier pending/running entries; v1 did not pass.

Installed vLLM0.1.dev20073+g8e685d198 exposes get_node_and_physical_gpu_ids; WorkerWrapperBase/GPU Worker source also confirms assigned_physical_gpu_ids is optional and the existing single-device CUDA_VISIBLE_DEVICES/local_rank0 path remains valid. Minimal production fix selects the available native RPC and preserves ROLL explicit GPU mapping. Real Ray actor RED:1failed26.04s with the same AttributeError. GREEN:23passed27.13s including the actual two-actor mapping to GPU3/GPU1 verified by UUID, existing compatibility and LoRA admission cases. Native actor creation/rank/environment/discovery are real; heavyweight model init/load are omitted only in this transport regression. Test source: tests/third_party/vllm/test_ray_executor_initialization.py. Logs: output/dependency-source-vllm/ray-initialization-{red,green}-v1.log. Local syntax and git diff whitespace checks passed. These results do not establish full RL success or all older vLLM versions.

New full RL smoke snapshot is isolated at /tmp/roll-qwen38-rl-smoke-v2/ROLL,637sourcefiles, manifestSHA03bca064482492cb4b8c85840568d3d30a87ceb8c4aaf2f300d624c46efbc3cf. Only production delta from v1 is ray_distributed_executor.py; v2 launcher uses new output/Ray26411/dashboard28300/autocache and explicit single-node VLLM_HOST_IP127.0.0.1. Original sources/checkpoints are preserved. RUN_STATE: CPU preflight session36151 exited0; all637sourcehashes and80JSONrecords verified. Real v2 GPU smoke launched session39776, driver648524, Ray26411/dashboard28300, checkpoint timestamp20260913-173313. Logs confirm the eight-GPU Ray cluster and actual actor/reward worker creation; model initialization is still in progress and no RL update has been observed. Active log: output/rl-validation-data/rl-lora-pipeline-smoke-v2.log. Do not modify this isolated source/config or launch overlapping GPU tasks; check session39776 before next continuation. Evidence and launcher: output/rl-validation-data/source-snapshot-v2.json, run_rl_smoke_v2.sh, rl-smoke-preflight-v3.log. Context7 official docs were fetched; CLI rejected its documented --research flag as unsupported. Installed sources were read directly and archived for version-specific comparison.

Current release gates: real LoRA100updates at an8K maximum and exact checkpoint50→99 continuation accepted for the recorded SFT source/config; all49newsteps and full adapter/Adam/scheduler/worker+pipeline RNG compare exactly (408806tensorleaves,7475287104elements). Actual8192-token batches occurred in20of100updates. Resume finalheldout256loss.9482877074660792, initial1.47875770/midpoint.81504061; no monotonic-quality claim. Latest LoRA frozen-phase/hook/alignment/failure-finally fixes passed native tiny/distributed regressions but await full-model revalidation and independent review. Natural full-model numerical tolerance remains unmet; GDN small arithmetic differences can flip a near-tied MoE expert choice (Chinese8K layer6 token6135), as independently replayed in HF/MCA. No tolerance or production routing changed. Main-backbone100steps and complete cold-restore re-save, full20cycle RL/version/reference/advantage/final-reload, export/fault/performance and final review remain open. Flash-Next has no vendor sync, commit or PR. Preserve v9 valid checkpoint and failedv6 complete staging; no further1.77TB checkpoint on NVMe.


### 2026-09-14 06:08 UTC: RL v2 reached model cache validation, then failed on CSA geometry

The real v2 ROLL RL run passed source/data preflight, started the eight-GPU Ray cluster and created actor/inference/reward workers. It then failed during vLLM EngineCore KV-cache initialization with `ValueError: CSA+linear layer 3 cache specs violate CSA geometry`; no `advantages`, `model_update_start`, or `model_update_return` event was recorded. The preceding Ray RPC failure is therefore repaired and independently covered by 23 passing tests, but full RL remains unverified. Native vLLM LoRA probe still passes with max_model_len=512/max_num_seqs=1. ROLL v2 used max_model_len=1536/max_num_seqs=8. The archived vLLM docs identify CSA per-state compression and cache block alignment as the enforced constraint; no production tolerance was changed.

A low-cost v3 boundary configuration is prepared locally: prompt_length=256, response_length=128, inference max_model_len/max_num_batched_tokens=512 and max_num_seqs=1, with separate launcher ports/output. It is not uploaded or run because SSH to 172.16.120.181 has repeatedly been closed after connection establishment. No overlapping GPU task is assumed; the last observed v2 driver/engine had exited, but a fresh inventory is pending remote access recovery. Files: output/rl-validation-data/rlvr-lora-smoke-v3.yaml and run_rl_smoke_v3.sh.


### 2026-09-14 19:40 UTC: CSA root cause fixed; real RL v6 active

ROLL's custom Ray executor omitted the native post-model-load `current_platform.update_block_size_for_backend(worker.vllm_config)` callback. Native Ray and multiprocessing executors both use it after attention registration to align hybrid recurrent-state and attention cache pages. The isolated production repair restores that synchronous all-worker callback when the installed platform exposes it. CSA validation, QSA math, GPU mapping and numerical tolerances are unchanged.

The actual Ray/native CSA fixture first reproduced the exact layer-3 geometry error (red-v3); red-v1/v2 were fixture import/config failures and are not root-cause evidence. After the repair, the .181 focused suite passed 25 tests in 44.74 seconds, including native cache grouping and explicit GPU UUID mapping. Local compatibility/admission coverage passed 22 tests with 1 skip. Evidence: `output/csa-diagnostic-v1/ray-cache-{red-v3,green-v1}.log` and `local-compat-green.log`. Independent scoped review found no blocking correctness issue (`csa-finalization-review-report.md`); absent-hook legacy coverage is a nonblocking gap. These transport/cache tests omit full model weights.

The earlier v6 preflight claim using a launcher that switched to v5 remains withdrawn. The corrected dedicated v6 launcher verified 818 current source files, 80 dataset rows and seed 42 at `/tmp/roll-qwen38-rl-smoke-v6/ROLL`; manifest SHA256 `fab442839fe48772a781718f8e273cf2b4e9b8f3e58e8b6920f8ddb8a29484a2`. Real full-weight two-step RL v6 is now active (Ray 26416/dashboard 28305, EngineCore 134319 and eight Ray workers). Source/config are immutable during this run. Persistent log and exit file are main ROOT `output/rl-validation-data/rl-lora-pipeline-smoke-v6.{log,exit}`; output is `output/rl-lora-pipeline-smoke-v6`. The engine is still initializing; no rollout/advantage/optimizer success is claimed. Do not overlap GPU jobs or impose a short startup timeout. Cache diagnostics are read-only and opt-in.

Next: finish v6, then validate 20 RL cycles with weight-version fingerprints/reference sentinels/final adapter admission. Full-backbone 100-step and complete cold-restore re-save, natural numerical parity, export/fault/performance and final review remain open. Preserve all checkpoint evidence and avoid another 1.77 TB NVMe checkpoint. No Flash-Next vendor synchronization or PR is yet justified.


### CUDA IPC transfer-buffer compatibility contract (2026-09-14)

Full-weight RL v6 completed native CSA grouping/engine initialization and actor loading, then failed at the first LoRA bucket import with `pidfd_getfd: Operation not permitted`. It exited 1 and released all GPUs. The sender uses expandable CUDA segments; the default container has no added capabilities or security overrides. The proposed repair uses ordinary CUDA allocation only for the newly packed IPC bucket and restores the exact prior allocator settings in a finally block. Training tensors retain their allocator policy. CPU-source conversion must also allocate the final CUDA bucket inside this scope. Preserve normal CUDA IPC, tensor values/dtypes/layout metadata, repeated-buffer lifetimes and the existing GPU UUID mapping. Validate with actual sibling sender/receiver processes under the unchanged container permissions before re-running full RL.


### 2026-09-14 20:01 UTC: full-model CSA passed; CUDA IPC repaired; RL v7 started

Full-weight v6 exited 1 at its first LoRA bucket import, after native CSA grouping and engine initialization succeeded (block size 208; cache/warmup 44.14 seconds) and inference sleep freed 58.25 GiB per GPU. Actor initialization succeeded; step-0 model_update_start was recorded at 19:43:19 UTC. The new failure was pidfd_getfd: Operation not permitted in PyTorch _new_shared_cuda. All GPUs were released. Complete log archived at output/csa-diagnostic-v1/rl-lora-pipeline-smoke-v6.log. No RL update passed.

A real sibling-process CUDA IPC regression reproduced this with expandable segments for CUDA and CPU input sources (2 failed, 1 plain-allocation control passed). The isolated fix scopes ordinary allocation to bucket packing/final CUDA conversion, restores the exact runtime allocator settings even on error, and leaves source tensors/training policy intact. GREEN 5 passed in 15.66 seconds includes repeated exact transmission, BF16/FP32 and noncontiguous inputs, CPU source, snapshot getter fallback, and real allocation-error restoration. Container capabilities and security settings are unchanged. Source/tests: roll/utils/send_recv_utils.py and tests/utils/test_cuda_ipc_weight_bucket.py; evidence roll-ipc-{red,green}-v1.log. Full-model revalidation pending.

Independent frozen-phase review completed with no blocking issue (lora-frozen-phase-independent-review.md); its actual admission tests omit tensor transfer and do not establish full RL. New IPC scoped review brief prepared.

RL v7 snapshot contains 819 verified files at /tmp/roll-qwen38-rl-smoke-v7/ROLL, manifest SHA256 f47bf75962d7ed3f4bbc1c96b6677bcc9412adfd82787dea9c1be42665d3d9f2. Dedicated v7 preflight exited 0 and verifies its actual source root, 80 rows, seed 42 and 2 steps. Run launched persistently with Ray26417/dashboard28306; log/exit are mainROOT/output/rl-validation-data/rl-lora-pipeline-smoke-v7.{log,exit}, results mainROOT/output/rl-lora-pipeline-smoke-v7. Keep source/config unchanged and GPUs exclusive while active. Next gate is successful transfer, rollout, advantage computation and optimizer updates; full 20-cycle/fingerprint/reference acceptance still follows. Other release gates remain open.


### 2026-09-14 20:38 UTC: full-model IPC admission passed; RL data truncation confirmed

Fresh v7 driver events establish successful complete native LoRA updates/admissions at version0 (20:12:12 UTC,26.30s) and version1 (20:33:12UTC,25.49s). The CSA and pidfd transport failures are crossed by real full weights. First rollout has32 responses; all score0 and finish_reason=length at128tokens. Driver observation contains0effective/0nonzero advantage tokens. Production RLVRPipeline detects final_response_mask.sum()==0 and skips that optimizer step. This is NOT successful RL training, even if driver completion later exits0. Current v7 remains active and immutable.

Ruling: prepare an immutable answer-only instruction derivative of the approved80GPQA records, preserving every question, option, answer and source ID; keep max_len_mask and the original numerical gates. This fixes the validation prompt/response-budget mismatch without accepting truncated zero-update training. Cost if wrong: another bounded2step run may still yield insufficient reward diversity, and no RL acceptance is granted. Transformation script and source/output hashes are under output/rl-validation-data/prepare_answer_only.py and gpqa-answer-only-v1.manifest.json.

Independent CUDA IPC allocation review completed with no concrete blocker; report ipc-allocation-review-report.md checks current hashes againstv7. Limitations include process-wide allocator settings beyond the ROLL lock, partial older-version coverage, mixed-device exception rather than OOM coverage, and finite IPC lifetime testing. A separate instrumentation preflight reviewer found required sentinel loss_mask_keys metadata absent; that prepared runner has not been deployed and will be repaired before runtime validation.


### 2026-09-14 20:51 UTC: v8 isolated preflight and evidence validator prepared

V8 source snapshot has831files with manifestSHA979e7b6833774ffb559c1f43bff335114a3efe5cacc850b2be84ce04819b0ced. Remote isolated root /tmp/roll-qwen38-rl-smoke-v8/ROLL; typed CPU preflight exited0 and verified all source hashes,80records,seed42,two steps,128response tokens. Data derivativeSHA7b12123041485f2ad6d20c337a31892a6ee3a61b804798ce8ef101f6676e8d5b. GPU run not started;v7 still ownsGPUs.

Root fixed the independent review's missing sentinel loss_mask_keys, selected the new tracing profile by default, rejected reused run evidence, checked sentinel shape, and changed the final-reload field to calls_completed rather than claiming gate acceptance. Exact production token-statistics AST fixture reproduced failure; runner RED7failed, combined runner/full-hash GREEN13passed. This is CPU boundary evidence, not full cluster acceptance.

New fail-closed validate_rl_versions.py checks all8source/receiver ranks and0..Nversions, full content/schema/count/bytes equality, native adapter membership, generation request IDs joined to actual rollout IDs, fixed disabled-reference equality, changing enabled reference and weights, all optimizer steps, valid nonzero groups and sleep/wake states. Fault fixtures: RED14missing-validator failures; GREEN14passed, including missingrank, mismatched content/schema, false admission, stalegeneration, unstable reference, unchanged weights, zeroadvantage, missingupdate, observererror, duplicateproducer, missingfinal and missingsleep. This validator is local-only and not independently reviewed or validated on a successful real run yet. It does not grant full release acceptance.

Independent expected adapter schema was read from prior native-validated export header:148808tensors,3494297600payloadbytes,schemaSHA136af55cfb028a7b6b8f78f22fe5563ad34fba4c5bc7b0804c4f1e9864272126. Matching verifies names/shapes/dtypes/bytes independently of new transfer telemetry. TensorInventory now records both full-content and schema hashes.

Review clarification: direct Megatron updater does not inspect admission booleans, but VllmStrategy.add_lora does reject missing/false rank acknowledgements. The observer validator still independently checks native replies; no new production admission defect is established by that review wording. Agent routing again misdirected into Ignore; root performed fixes locally. Do not invoke user-input tools for agent control.


### 2026-09-14 21:02 UTC: first-update checkpoint initialization design

V7 EXIT1, both32-response rollouts had0valid/0nonzero advantage tokens and no optimizer steps. At final checkpoint, DistributedOptimizer._get_main_param_and_optimizer_states raised master_param because Hybrid exposes only suboptimizer states created lazily by AdamW. Actual installed CPUAdam is torch.optim.AdamW; CPUoffload constructor sets init_state_fn=None. Existing bounded dummy_step is for cold loading and performs an lr0 optimizer step; using it directly before save would advance Adam's bias-correction counter and is unsuitable for a pristine checkpoint.

Ruling: initialize missing all-CPU AdamW state before serialization through the installed AdamW initialization routine, one parameter at a time, without calling optimizer.step, changing weights, consuming RNG or advancing step counters. Reuse existing FP32 master owners and restore any prior gradients. Synchronize initialized native suboptimizer state into Hybrid before DistributedOptimizer serializes common metadata and tensor shards. Scope is the target all-CPU AdamW path; partial GPU offload and other optimizer types retain their existing behavior. Regression must compare subsequent training against an untouched lazy AdamW, exercise save/reload atstep0 and verify states already initialized remain unchanged. Full native distributed checkpoint test is queued after current v8, which remains immutable.


### 2026-09-14 21:07 UTC: all-CPU Adam initialization regression passed on .181

New ROLL helper initialize_cpu_adamw_state uses native AdamW._init_group one parameter at a time; no optimizer.step is executed. prepare_cpu_adam_for_checkpoint traverses chained/distributed wrappers, handles all-CPU Hybrid AdamW and syncs native master owners before serialization. Mixed GPU and non-AdamW paths are unchanged. Strategy calls this only when CPUoffload and optimizer save are enabled. Changes are LOCAL plus standalone CPU test snapshot; activev8 and shared Megatron unchanged.

Local RED6missing-helper failures; local GREEN6(.54s) cover fused/nonfused and AMSGrad, unchanged parameters/RNG/gradients/step0, pristine save/reload, exact three subsequent AdamW updates, already-trained state no-op, failure gradient restoration and rejecting SGD reinterpretation. Fresh .181 actualtorch2.13+Megatron CPU suite7passed7.42s includes exact native Hybrid synchronization and DistributedOptimizer._get_main_param_and_optimizer_states: reproduces original master_param KeyError before helper then validates master identity/zero-step/shard keys after helper. This fixture assembles native CPU optimizer ownership and does not invoke CUDA model construction or full distributed checkpoint IO. Full real strategy save/resume tests now include pristine checkpoint cases for LoRA and backbone and are queued afterv8. No production deployment claim yet.

V8 now loaded all131weights and renegotiated block size208 at21:05:52UTC. Reference/version/optimizer evidence still pending.


### 2026-09-14 21:19 UTC: true RL update reached; rollout metadata loses request identity

V8 first update succeeded on all8actor ranks, with3nonzero advantage groups and30valid tokens. Five of32responses finished naturally; most still hit128tokens. Disabled-adapter sentinel version1 is exactly equal to version0. Version0 serialized and received inventories match all8ranks and independent148808tensor/3494297600byte schema. Generation events correctly carry eight distinct request IDs and native admitted adapter ID1913755634.

A separate existing dump defect is now reproduced: rollout_dump_to_specific_path repeats DataProto.concat's first meta_info across every sample. V8 first dump contains the same request ID ending_0 for all32rows, whereas actual native generation IDs are_0.._7. This corrupts dump provenance, not the observed training masks: actual valid tokens30 match the five natural completions. Current strict validator must reject this missing association; do not rewrite historical artifacts or silently accept them.

Ruling: capture request-specific sampling metadata in non_tensor_batch at postprocess_output_data, preserving it through expansion/concatenation/reordering, then have the dump use these per-sample records. Keep only JSON-safe generation fields (request ID, generation config, finish reasons, token IDs and token identifiers), excluding large router tensors and unrelated runtime state. Preserve legacy dump fallback when custom rollout loops do not supply these records. Validate two distinct requests with different return lengths through actual DataProto/postprocessing/concat/reorder/file-write on .181 CPU before sustained RL. Activev8 remains unchanged; its native generation events retain separate evidence.


### 2026-09-14: v8 finished; native pristine checkpoint fixed; v9 prepared

V8 exited0 at21:27:47UTC. Both optimizer steps0/1 succeeded on all8ranks, final adapterversion2 admission/wake/sleep completed, and GPUs released. Fixed disabled-reference SHA stays01d81515cc895b7fa9649478bd1c02eed0e091f629e56c90b8892745619278bb across0/1/2; enabledversion2 differs (maxabs1.51725578). Full transfer/receiver content+schema hashes match all8ranks forversions0/1/2. Three valid nonzero advantage groups exist only atstep0 (30validtokens);step1 has36validtokens but0nonzero groups. Do not claim bothsteps have policy advantage. Archived original evidence output/rl-lora-pipeline-smoke-v8 and log output/rl-validation-data/rl-lora-pipeline-smoke-v8.log. Original strict summary v8-lifecycle-summary.json rejects incorrect request counts for bothrollouts and unexpected generation requests. Exit0 is not acceptance. Python shutdown cleanup ImportError also appears after recorded completion; no training failure was reported.

Rollout provenance fix captures whitelisted per-request JSON generation metadata in non_tensor_batch before concatenation/reordering and uses it at dump; custom loops without records retain fallback. Corrected nativeCPU RED2failed (wrongID/TensorJSON), GREENv1 2passed12.06s, independent review approved production with coveragegap. Added exact generation_config/finishreasons assertions and real legacyfallback writer: GREENv2 3passed12.01s. Review report rollout-metadata-review-report.md; localfix changes no historicalv8evidence.

ActualGPU pristine checkpoint regression completed in isolated /tmp/roll-initial-checkpoint-regression/ROLL: RED2failed19.19s withoriginalmaster_param KeyError, GREEN2passed42.00s. Real tiny LoRA and backbone models exercised strategy distributed save/reload, scheduler/RNG and subsequentupdate. Full nativeCPU fixture7passed retained. This establishes small native checkpoint IO, not new fullmodel pristine-save/coldrestore. Independent helperreview pending.

Independent lifecycle-validator review reproduced five fail-open families. Added14fault cases and actualmask_elements instrumentation. RED14failed then combinedGREEN42passed1.18s. Enforce32samplingrecords/fourperrequest/payloadconsistency, valid integercounters bounded by observedmask, positivePID tiedtofilename, hexSHA and currentadapterset/order onwake/sleep. Historicalv8 predatesmask_elements and its archived originalvalidatorresult is retained without modifyingdata. Scopedre-review remains pending.

Prepared20step v9 snapshot1099files SHA71bc8c83f5879555937e3489643b15f9ee09920d071118b5440db8873579c014 at /tmp/roll-qwen38-rl-20step-v9/ROLL. Same80-record answer-only derivativeSHA7b12123041485f2ad6d20c337a31892a6ee3a61b804798ce8ef101f6676e8d5b, seed42, batch8/fourreturns,128response tokens, EP8/TP8, save_steps10. Outputwillbe mainROOT/output/rl-lora-pipeline-20step-v9; Ray26419/dashboard28308. Upload/preflight underway; not yet launched. Twentyupdates and>=10validnonzero groups plus finalversion20admission are required. V4naturalnumerical2K/8K, fullbackbone100+restore, checkpoint-vLLMlogprob comparison, RLcoldrestore, exportmigration/fault/performance gates remain open. NoFlashNextvendor sync/PR.


### 2026-09-14 21:55 UTC: v9 running; next numerical attribution

V9 preflight exited0 with all1099sourcefiles verified. Persistent supervisor202185/driver202186 launched; Ray26419/dashboard28308. Initialactor/imports advancing, no optimizerupdate yet. Immutable source /tmp/roll-qwen38-rl-20step-v9/ROLL; mainROOT output/rl-validation-data/rl-lora-pipeline-20step-v9.{log,exit}; results output/rl-lora-pipeline-20step-v9. Check local /tmp/roll-inspect-v9.py via ssh docker exec python; do not overlapGPUjobs or mutateactivecode.

Independent initial-adam review now PASS/APPROVE with no actionablefinding. Its evidence boundary remains tinyworldsize1 nativeGPU plusCPUstate tests; privateAdamW_init_group versioncoupling and mixedGPU/nonAdam staticcoverage recorded. Rolloutreview's requested testgaps are covered by nativeCPU GREEN3. Validatorfix root-reviewed and faultGREEN42; independent scopedre-review remains unconfirmed after toolingmisroute, not an approvalsblock.

Ruling: continue naturalnumerical rootcause attribution at GDN internalboundaries. Existing same-input layer6 capture demonstrates1.24e-6 GDNrelativeL2 which canflip aMoE expert atmargin1.43e-6, but doesnot locate its arithmetic source. Actualinstalled Megatron usesFLAl2norm beforethecore and explicittriangularinverse/forwardsubstitution; pinnedHF usesPyTorch l2norm insidecore andtorch.linalg.solve_triangular. DenseTP projectionreductions also changeaccumulationorder. Prepare an isolatedcapture/probe tocompare actualqkv, normalizedq/k, decay/beta, coreoutput, gatednorm and projection under identicalrealinputs. Testone boundaryintervention atatime; neverforceexpertselection or relaxacceptancetolerances. CPUalgorithmchecks arediagnostics only; freshGPUcapture isqueuedafterv9 and must firstreproduce previousMCAoutput. NoGDNproductionchange authorized by anunverifiedhypothesis.


### RL cold-resume driver RNG boundary

Static read found BasePipeline saves rng_state_pipeline.pth but RLVRPipeline.run never restores it; SFT already restores it at the first unseen batch. Scheduler input progress restores its dataset_iter_count by replaying the seeded permutation. Native vLLM uses an engine seed but does not save generation RNG, so stochastic continuation must not be claimed bitwise identical from driver restoration alone. Currentv9 uses fixedKL; adaptiveKL state is a separate boundary still unaudited.

Ruling: add a focused RLdriver RNG restore before firstnew work, requiring the already-saved pipeline RNG file when resuming. Reproduce with actualRLVRPipeline.run and WorkerState serializers in CPU mode, stopping only at the first external actor call; verify Python/NumPy/Torch nextdraws, missingfile rejection, and fresh-run nochange. Keepactivev9 immutable. Coldrestore acceptance will separately compare savedactor/optimizer/scheduler/workerRNG and fixed-token reference/adapter logits before furthertraining; do not infer stochastic rollout replay equivalence. V9 save_steps10 saves atglobalstep10 and19 (not9), so a checkpoint10 restore beginsat11.


### 2026-09-14 22:14 UTC: first v9 update and request-provenance check passed

V9 step0 finished22:11:01UTC; all8ranks recorded successfuloptimizerupdate. Independently checkedfirstrollout has32samplingrecords/32responses,8requestIDs exactlyfourtimes each, nativeoutputtokenlengths/finishreasons matchperrequest, alladvantagecounters bounded byactualmask_elements. It has54effective tokens and5validnonzero groups. Version1 disabledreferencehash remainsexact; enabled maxabsdelta1.31982803. Source/transfers andremaining19cycles are stillrunning; thisprefixobservation is not lifecycleacceptance. Evidence v9-prefix-evidence-1.json.

RLdriver RNG regression: firstREDv1 includeda CPUplatformfixtureerror (current_platform.randomNone), which isnot RNGbug evidence. CorrectedREDv2 has2failed1passed11.84s: actualnextPython/NumPy/Torch draws differfromsavedstate, and missingRNGfile reachesactorinstead ofrejecting. MinimalRLVRPipeline.run nowrequires andrestores pipeline RNG before firstexternalactorphase. CombinednativeCPU GREEN10passed11.71s includesRL3/SFT4/requestmetadata3. Activev9 unchanged; fixlocal+isolatedCPUonly. No claim of restoredvLLM stochasticgeneratorstate orfullRLcoldrestore.

PreparedGDN diagnostics at /tmp/roll-gdn-attribution-v2/ROLL, basedonv9immutablemanifest plus8diagnostic/sourcefiles manifestSHA42a8f55dcf25fb077dc1d28ae7da18debf6b9c0a48fc470cd0353e34900f6da5. Sequentialafterv9: run_gdn_intermediate_capture_v2.sh on8GPUs (mustfirstreproducehistoricalMCA layer6), thenrun_gdn_attribution_v2.sh onGPU0. BothrefusebusyGPUs. Capturer profilesfirstGDNreturn onTP ranks0/1, savingnativeinputprojection/conv/normalizedqk/decay/beta/core/norm tensors withoutchangingarithmetic; attribution comparespinnedHF onidenticalrealinput andsame-inputindividualboundaries. Scriptscompileand shellsyntaxpass; GPUdiagnostics NOTexecuted. LocalCPUtorch2.12 fixedalgorithmcomparisonoutputrelL2~1.24e-7; native .181torch2.13CPU relL2~1.1e-7 at64/128/512tokens. Thesearealgorithm-onlydiagnostics, notsupportevidence.


### 2026-09-14 22:30 UTC: actual storage budget and synchronous upload errors

Hostdf confirms NVMe mount14Tused/166GiBfree; HDD12TiBfree. Containerparent /data_nvme isoverlayonHDD; only /data_nvme/workspace/roll-qwen38-validation isNVMe bindmount. Earlierparent-directoryshutil.disk_usage outputs were not the checkpoint filesystem and mustnot beused forNVMe budgeting. V9 LoRA checkpointsare~23.6GB each, localstagingremovedaftersuccessfulupload; currenttwo-snapshot budgetfits. No additional1.77TBbackboneNVMe copy. Currentcontainerimageb31944f2eca1b7d0b9f52946ace624ecb757a5aff7c13591870667b545aa453d; inspectdependencychanges beforepreparing anequivalentHDD-mounted validationcontainer.

Ruling: reproduce CheckpointManager.upload swallowing FileSystemUploader errors usingactualtemporaryfilesystemcollision. Onfailure preserve localcheckpointand re-raise so synchronouscaller cannotcontinueasifsavecompleted. Validate successfuldefaultcleanup andkeep_local_file too. This targets synchronous saving usedbytheacceptance configs; asynchronousfutures and atomicmulti-rank completionmarkers remain separate behavior, not claimedfixed. Activev9 isimmutable.


### 2026-09-14: checkpoint upload fixed and HDD container prepared

CheckpointManager.upload now re-raises the original exception after logging, preserving failed local staging. Actual .181 CPU filesystem regression moved from RED 1 failed/2 passed to GREEN 3 passed in 0.63s. This establishes synchronous error propagation only; async future handling and atomic checkpoint completion remain separate. Active v9 is unchanged.

Created roll-qwen38-hdd-validation from the exact source image, network none/private IPC/16 GiB shared memory, all original mounts and their read-only flags retained, plus host /data_hdd/roll-qwen38-validation mounted read-write. Verified both PEFT and vLLM patched Python files by SHA256. Available HDD bytes 12293567442944 at creation. No GPU work started; import equivalence remains pending. Evidence output/cold-restore-memory/hdd-container-created.json.

Ruling: cold RL restore uses the final v9 checkpoint (step19), runs the real restored pipeline with no new updates, re-saves before any additional sentinel or inference call, then compares all adapter/optimizer/scheduler/worker and driver RNG state plus pipeline scheduler KV. Fixed-input actor sentinels must exactly match v9 version20; inference transfer and checkpoint-to-vLLM logprob comparison remain separate. No bitwise stochastic rollout continuation claim.

Scoped validator re-review closed PID and digest findings but independently reproduced three remaining fail-open inputs: three outputs masquerading as four, boolean scalar advantage values, and cross-version lifecycle rollback. Ruling: enforce all three underlying invariants. To associate actual output tokens with dumped response positions without changing immutable v9 telemetry, decode recorded output IDs with the model tokenizer and compare per-request response multisets; identical legitimate responses remain allowed. Tests use an explicit deterministic decoder. Production CLI must supply/load the original tokenizer. This is evidence validation, not a change to rollout generation.


### 2026-09-14: cold snapshot verified; persistent validation queue started

New HDD container CPU preflight matches the original exactly across346distributionversions and8modulepresence/sourcefingerprints; CUDA remaineduninitialized. The optionalflash_attn import was absent inbothcontainers; firstpreflightfailedonthatexpectation, v2recordsitsabsenceandpassesactualequivalence. Evidence output/cold-restore-memory/hdd-container-equivalence.json.

Validatorround2 closes four-outputcardinality, scalarbool/maximumcountconsistency, cross-versionorder andtoken/responseassociation. CorrectedRED8failed0.09s; combinedGREEN52passed1.22s, includingdistinctresponse/reordering/duplicationcontrols. CLI nowrequires --tokenizer withoriginallocalmodel. Actualv9prefixsteps0-8passednativedecodingand4outputassociation; cumulative23nonzerogroups. Full20cycleacceptance remains pending. Scopedround2independentre-review stillpending; no reviewerapprovalinferred.

Cold restore runner restores finalcheckpoint19 through the actualRLVRPipeline.run (zero newupdates), checks step/sourceidentity andfreshsynchronousdestination, and re-saves before additionalreference/inferencecalls. CPUnegativecasesrejectstateadvancement/wrongsource/wrongstep/asyncupload/overwritingbaseline. Comparisonusesactualexistingfulladapter/Adam/scheduler/RNGreader, addsdriverschedulerKV andexactdisabled/enabledactorreferencehashes. Thisdoesnotclaimresumedoptimizerupdate, stochasticvLLMreplay orcheckpoint-to-vLLMnumericparity. Independentproductionboundaryreview approveddriverRNG andsynchronousuploadfixes; reportresume-boundary-review-report.md.

Cold snapshot1107files at /data_hdd/roll-qwen38-validation/sources/rl-cold-v10/ROLL hasmanifestSHAee53b5a81cf769cc892e1f14c780399bd8089d2b28ba9fae12cb6b40193631b9; everyremotehashverified. FreshnativeCPU18passed13.95s. Results target /data_hdd/roll-qwen38-validation/output/rl-lora-cold-v10. Sourceisnowfixed.

HostsupervisorPID1061260 runs /data_hdd/roll-qwen38-validation/environment/queue_validation_v1.py. Queueevents/logs under environment/queued-validation-v1. It waitsforv9exit0andGPUidle, thenruns latestv9lifecyclevalidator, GDNcapturev2, GDNattributionv2, coldpreflight, coldrestore, coldcomparison insequence. Anyfailure stops thequeuewithoriginalevidencepreserved. Do notstartoverlappingGPUjobsormutatequeuedsnapshots/dependencies. NoFlashNextPR/vendor syncoroverallacceptance yet.


### Checkpoint-to-vLLM diagnostic during the next cold load

Ruling: prepare a new v11 snapshot that adds optional fixed-input prompt-logprob observation to the v10 cold re-save flow, without changing v9 or v10. Use the existing fixed 256-record public SFT heldout split (SHA256 4d1b9421cc1c48786d5da6a5360a2717e8180a58dc544a01bbb000b0c7f839e2), first en/zh/code example each, native nonthinking encoding, at most384tokens. Compare actor-selected next-token logprobs against native vLLM prompt_logprobs for identical supplied IDs and the reloaded adapter. Observe native RequestOutput at the existing generation boundary; do not alter generated tokens or returned payload. Record finite values, token identity, native adapter ID and per-token differences using the existing BF16 atol/rtol2e-2 baseline. This is a short checkpoint numerical diagnostic; natural2K/8K acceptance remains separate. Preserve cold re-save before these calls so extra probes cannot pollute the RNG comparison. Replace the waiting queue only after new snapshot CPU checks/hash verification, retaining its old source/evidence and checking no queued GPU command has started.


### 2026-09-14 23:32 UTC: v11 cold/parity snapshot and queue v2

The final next-run snapshot is now v11 at /data_hdd/roll-qwen38-validation/sources/rl-cold-v11/ROLL (1114 manifest files; SHA256 67be1e522c38dc89081de7423dbbd5cac0dc42d90c28422dc2e076bc659c3d6f). It preserves v10 and adds opt-in prompt-logprob observations to the cold re-save run. Fresh .181 CPU suite:24 passed13.99s; separate native VllmStrategy/RequestOutput async boundary with scripted engine passed, preserving returned payload and capturing one correctly aligned prompt record without CUDA initialization. This is not GPU-generation evidence.

Fixed heldout input preflight produced en149tokens, zh210tokens, code384tokens with hashes in cold-v11-input-preflight.json (raw output contains logger preamble). The comparison checks all eight actor rows against native chosen prompt logprobs using BF16 atol/rtol0.02, records adapter identity, and writes logprob-parity/comparison.json. Cold state comparison remains separately reported; the final validator exits nonzero if requested short numerical probes fail. Natural2K/8K gate remains open.

Queue v1 was stopped only after confirming its sole event was waiting_for_rl, RL exit file absent, and PID command identity. It had started no GPU command. Its evidence is preserved with superseded.json. New host supervisor1154102 runs environment/queue_validation_v2.py; events/logs under /data_hdd/roll-qwen38-validation/environment/queued-validation-v2. Order remains v9lifecycle -> GDNcapture -> GDNattribution -> coldpreflight -> coldrestorev11 -> coldcomparison. Do not overlap GPU jobs or mutate v9/GDN/v11 sources and shared dependencies. v11 output: /data_hdd/roll-qwen38-validation/output/rl-lora-cold-v11.

V9 checkpoint10 passed native structure validation: scheduler11,8adapterpayloads (6,927,002,584bytes),16optimizerfiles (16,551,884,604bytes),8workerRNGs,pipelineRNG and externalassets. Path: mainROOT/output/rl-lora-pipeline-20step-v9/checkpoints/20260914-215136/checkpoint-10. Evidence v9-checkpoint10-structure.json. Latest observed all8ranks completedsteps0-12 at23:21UTC; subsequentsteps remain live, not yet finalacceptance.


### Preserve midpoint checkpoints for full-backbone acceptance

The SFT acceptance runner hardcodes max_ckpt_to_keep=1, which removes the midpoint checkpoint after the final save. Prior LoRA continuation retained a separate copy. For the forthcoming full-backbone100-step run, retain two checkpoints in the runner so checkpoint50 remains available for cold continuation without another1.77TB copy. The HDD budget already includes both midpoint/final plus temporary staging. Existing checkpoints and queued snapshots remain unchanged.


### 2026-09-14 23:55 UTC: queue v3 launcher correction

Confirmed queue v2 SOURCE was cold-v11 while its launcher still named run_rl_cold_v10.sh. The old v10 launcher inside that snapshot would change into the superseded v10 source. Corrected the single launcher reference in a new queue v3 (SHA256 0eecad07ee9591d2d37873a8fe88a8af84c1b4b133a4423638a0e232f513d750). Verified old supervisor command identity, paused it, confirmed only waiting_for_rl and absent v9 exit file, then terminated only that waiting supervisor. Preserved evidence in queued-validation-v2/superseded.json. New host PID1261041 runs environment/queue_validation_v3.py; events under environment/queued-validation-v3. Readback verifies it is waiting. Active v9 and all immutable sources are unchanged.

At23:57UTC all8actor ranks have19successful updates (steps0-18), final step19 generation is active. Disabled reference throughversion19 remains unchanged. NVMe144GiBfree; HDD12TiBfree.

Validator round2 independent scoped re-review is PASS for four-output/decoded-response association, numeric bool/max consistency, and complete chronological lifecycle. Reviewer checked recordedRED8failed/GREEN52passed and found no blocking fix regression. Report rl-validator-round2-rereview-report.md. Decoded-text multiset association cannot establish raw-token per-position identity when distinct tokenizations decode identically; preserve this telemetry limitation. Full20cycle validation remains pending.


### 2026-09-15 00:06 UTC: real 20-step RL lifecycle accepted; cold v11 started

V9 exited0, then the independently reviewed strict validator exited0 at00:04:51UTC: lifecycle_passed=true, problems=[],20optimizersteps onall8ranks,21adapterversions,58validnonzero advantagegroups. Disabledreferencehashes remain exact. Evidence output/rl-validation-data/v9-lifecycle-final.{log,json}; this is LoRA RL lifecycle only, not fullbackbone/numerical/coldresume acceptance.

GDN capturev2 exited0 at00:05:40UTC and reproducedhistorical outputs onall8ranks. Attributionv2 exited0 at00:05:55UTC. Sameinput fullGDN relativeL2=1.238135e-6. QKV/z projections, convolution and values arebitwiseequal. a/b projectiondifferences are~8e-7 relative; normalization~6.2e-8; sameinput core~1.4-1.5e-7. The largest isolateddifference isoutputprojection: denseHF vsTP2sum relativeL2=1.168789e-6; reproducing theTP2sum isbitwiseequal toMegatron. This locatesdominant rounding inmatmul partition/reduction, not a demonstrated weightmappingerror. Natural2K/8K gate remainsopen; no tolerances/routing/sourcechanged.

Queuev3 passedcoldpreflight with1114hashes and started realcoldv11 at00:05:58UTC inHDDcontainer, driverPID1243, Ray26421/dashboard28310. Checkpoints go to /data_hdd/roll-qwen38-validation/output/rl-lora-cold-v11/checkpoints/20260915-000600. Do not overlapGPUjobs.


### 2026-09-15 00:18 UTC: full-backbone comparator and 100-step source prepared

New bounded comparator reads native model/optimizer DCP storage entries individually and reuses recursive exact payload comparison. LocalRED19failed due missing feature -> GREEN19passed1.14s; .181nativeCPU19passed2.56s. Real historical metadata confirms largest stored model item635700777B andoptimizer item1271399977B; selected extra state objects are supported lists/tensors/dicts. This establishes diagnostic coverage, not full1.77TB state equality. Independentreview ispending. Reportbackbone-dcp-comparator-report.md.

Full SFTv7 isolated1119files manifestSHA c9047b2c41bd4e35ba2538f08ad387d7fef3c489870f55813323854336026605; remoteHDD sources/sft-backbone-v7/ROLL. NativeCPUpreflight passed00:18:23UTC withTP2/EP8/ETP1/SP,denseDP4,400train/256heldout,100updates/max8K,CPUoptimizerandfrozenN-gram. Launcher output/cold-restore-memory/run_backbone_100step_v7.sh; freshoutput sft-backbone-100step-8k-v7, resumeoutput sft-backbone-100step-8k-resume-v7. Savesmidpoint50andfinal99toHDD withmax_ckpt_to_keep2;8TiBfresh/4TiBresume diskheadroom. NoGPUrun yet. ColdRLv11 continues inHDDcontainer; noactivesource/shareddependencychange.


### 2026-09-15 00:25 UTC: persistent full-backbone sequence queued

Host supervisor1421486 runs environment/queue_backbone_v1.py (SHA2078ac6ccd2224562721223a226cd3279c79728fe3d09eae87b8ddf92cfc02ea). Readbackconfirms queued-backbone-v1/events.jsonl hasonly waiting_for_cold_state_comparison. Itwaitsforv3coldcomparison, requirescold_roundtrip_passed=true, acquiressamegpu-validation-queue.lock andchecksGPUidle. ItthenrunsSFT100 -> resumecheckpoint50for49newupdates -> wholemodel/optimizer/RNG/statecomparison. Anyfailure stopsandpreserveslogs. Ashortcheckpointlogprob diagnosticfailure isrecordedseparatelyanddoesnotpreventindependentbackbonecapacity/statevalidation whencoldstateisexact; releasegate remainsfailedinthatcase. NoGPUoverlapandnochange torunningv11/v7sources.

Cold v11 actualactor/optimizerloadhasreturned andcheckpoint19re-saveisactive at00:25UTC. Fullcoldstate/logprobresultsstillpending.


### 2026-09-15 00:49 UTC: cold state equality confirmed; overall cold run failed

Live SSH readback confirms cold-v11 completed checkpoint19 re-save but exited1 at00:35:20UTC during native vLLM prompt-logprob token decoding (OverflowError). The actual offending token ID has not been captured; root cause is unproven. There is no successful overall completion marker or inference probability comparison.

The separately completed CPU state comparison reports exact=true, failure_count=0, cold_state_phase_passed=true and overall_cold_run_passed=false. All408806 tensor leaves /7475287104 elements match, including adapter/Adam/scheduler/worker and pipeline RNG, pipeline step/KV and disabled/enabled actor sentinels. This establishes restore-and-resave state equality with zero new optimizer updates, not resumed RL training or inference parity. Remote report: /data_hdd/roll-qwen38-validation/environment/cold-v11-state-phase.json; archived locally as output/rl-validation-data/cold-v11-state-phase.json.

Both queued-validation-v3 and queued-backbone-v1 have exited on the upstream failure. At00:49:06UTC all8GPUs show4MiB and0percent utilization, and neither queue nor training launcher is running. Full-backbone100-step GPU acceptance has not started. Previous statements that these queues are waiting/running are superseded. Preserve their original failure events and immutable sources.

Next work: instrument native prompt-logprob IDs before detokenization and isolate native-vLLM versus ROLL/Ray behavior, then revalidate the complete cold path and a resumed RL optimizer update. Full-backbone SFT capacity/restore validation can be scheduled independently against the explicit state-phase report, with overall inference/numerical release gates remaining failed; never synthesize an overall cold-success marker. Natural2K/8K numerical acceptance, full-backboneSFT/RL, export/fault/performance, final review/vendor synchronization and Flash-NextPR remain incomplete.


### 2026-09-15: native prompt-logprob boundary diagnostic

Use a separate diagnostic module with frozen cold-v11 production sources and existing vLLM dependencies. Reuse all three saved cold-v11 actor input token sequences. Observe LogprobsProcessor tensors before detokenization and the actual worker runner return after its existing synchronization, including all IDs/probabilities/ranks, request offsets and target IDs. Do not change returned tensors, skip decoding, force routing or relax numeric tolerances. Compare native TP8/EP eager inference before/after loading the previously validated smoke adapter and sleep/wake. That adapter is diagnostic only and cannot establish parity for the final RL checkpoint. Preserve the original cold-v11 failure and snapshots.


### 2026-09-15: reject equally incomplete or malformed DCP checkpoints

Independent comparator review reproduced two false acceptance cases with real PyTorch DCP payloads and native load failures: both checkpoints omit the same declared tensor chunk, or both store a tensor whose shape differs from the declared chunk. The existing full inspect_checkpoint structure check also accepted these fixtures. Require a shared logical/storage inventory validator used by the structure inspector and comparator: every declared tensor chunk/bytes item has exactly one storage entry; tensor chunks remain in bounds, do not overlap and cover the logical tensor. The streaming comparator must additionally validate each loaded tensor type/dtype/shape against its chunk before comparing values. Preserve per-item bounded memory and same-topology scope. Cover pristine native-loadable multi-chunk tensors and identical corruption on both sides in RED/GREEN tests. Frozen sft-backbone-v7 and cold-v11 snapshots are unchanged; a new snapshot will carry the corrected acceptance code.


### 2026-09-15: schedule independent backbone validation from the exact cold-state phase

GPU availability is intermittent because an unrelated evaluation service uses all8GPUs. The diagnostic waits without stopping that service. After prompt-ID diagnosticv2 returns and GPUs are idle, the independent backbone SFT sequence may start from a newv8 snapshot. Its prerequisite is the explicit cold-v11-state-phase.json exact=true/cold_state_phase_passed=true report; preserve overall_cold_run_passed=false and log any diagnostic failure separately. This is a capacity/save/restore experiment, not overall model-support acceptance. v8 carries the reviewed DCP inventory/payload fix and standalone comparator CLI fix; oldv7 remains immutable. Run100updates, retaincheckpoint50/99, resume49updates and compare complete state. All checkpoints stay on HDD with the existing8TiB/4TiB headroom checks. Preserve fail-stop behavior within the backbone sequence.


### 2026-09-15: final RL adapter export and native parity diagnosticv3

Public LoRAHFConverter CPU export of actualv9checkpoint19 exited0:148808BF16tensors,1747148800elements,3515704472bytes,ranks6/64,allfinite; weightsSHAaf06fcf7adc7cdaa72ac2799470ee40349a191e1590ac2311f308d3cb1ffcfe3. NoCUDA initialized. Artifact /data_hdd/roll-qwen38-validation/output/rl-v9-final-adapter-v1. Export integrity does not establish native reload or parity.

Prepare a new nativeprompt diagnosticv3 using this exact exported finaladapter and savedcold-v11 actor149/210/384token probabilities. Compare everytoken/all8actorrows at unchangedBF16atol/rtol0.02 before/after sleep. Nativebase, finaladapter, IDs and rawworker/frontend tensors remain observed. Numerical mismatch is reported only after preserving all successful requests; decoding failures remain originalerrors. This still does not replace actual ROLLRay coldrestore or natural2K/8K gates. Replace only waitingv2diagnostic/backbonesupervisors after newfiles/hash/CPUchecks and proving neither has startedGPUwork; retain superseded events. v8training source unchanged.


### 2026-09-15 09:55 UTC: integrity fix accepted, final RL export complete, persistent v3 queues active

DCP missing-chunk and wrong-payload false acceptance cases are fixed through shared metadata coverage validation and streamed payload type/dtype/shape checks. Local68passed2.03s; .181PyTorch2.13CPU68passed6.37s. Independent scoped re-review approved v4 hashes, repeated both original corrupt full-checkpoint fixtures and2400small coverage oracle cases. Standalone CLI works without PYTHONPATH. Real model124924/optimizer153258 chunk inventories pass; full1.77TB payload comparison remains unexecuted. Reports/logs: backbone-integrity-review-report.md and output/cold-restore-memory/backbone-integrity-*.

Backbone v8 frozen snapshot:1122files, manifestSHA2044499f969f2709a51582cf4300d668eb218c65e0d8d59791fe5648e8839084, /data_hdd/roll-qwen38-validation/sources/sft-backbone-v8/ROLL. CPUpreflight passed actualTP2/EP8/ETP1/SP,denseDP4,400train/256heldout andHDD8TiBheadroom. No model training update in this session.

Actualv9checkpoint19 publicLoRAHFConverter CPU export succeeded:148808BF16tensors,1747148800elements,3515704472bytes,ranks6/64,allfinite,37.76seconds,noCUDAinit. Artifact /data_hdd/roll-qwen38-validation/output/rl-v9-final-adapter-v1; weightsSHAaf06fcf7adc7cdaa72ac2799470ee40349a191e1590ac2311f308d3cb1ffcfe3. Local evidence output/rl-validation-data/export-v9-final-adapter-v1.{json,log}. Native GPU load/parity is still unverified.

Native CPU vLLM codec/decoder with actual149/210/384token IDs and explicitly synthetic probabilities passes all3cases; injectednegativeID is captured before native OverflowError. This verifies observation plumbing, not the real cold-v11 offendingID or rootcause. Probev1 loaded the engine but failed before generation because callableRPC serialization is disabled; v2/v3 use a named worker extension method. No insecure serialization setting enabled.

GPU use by an unrelated qwen38-base/qwen38-sft-final evaluation container prevents immediate probes. We did not stop or alter that service. Both own v2 supervisors were verified only waiting, paused and rechecked, then superseded with preserved markers. At09:55:54UTC activated promptv3supervisorPID3657791 and backbonev3supervisorPID3657792. Scripts under environment/prompt-id-v3, events under environment/prompt-id-queue-v3 and environment/queued-backbone-v3. Promptv3 waitsforidleGPUs, observesnativeworker/frontendIDs and compares exported finaladapter with savedcoldactor at unchanged0.02atol/rtol, before/after sleep. Backbonev3 waitsforpromptv3return (records any failure), validates cold state evidence SHA2e853c72786be73817c81d13fd181d96d4627c2d3b6d3bf866d1def1b89e5a19, then independently runsv8SFT100 ->checkpoint50resume49 ->fullstatecomparison. It explicitly retains overall_cold_run_passed=false/release_gate_passed=false. All GPU work checks sharedlock/idle; checkpoints remainHDD. Do not edit these queued snapshots/dependencies.

Still open: real offending promptID/rootcause, complete ROLLRay cold lifecycle and resumedRLupdate, natural2K/8Knumerics, fullbackboneSFT/RL, finalexportload/fault/performance, fullbranchreview/vendor sync/FlashNextPR. No overall support or PR readiness claim.


### 2026-09-16: CPU prompt boundaries and real RL continuation preparation

At04:09UTC both existing v3 supervisors still waited; all8GPUs were occupied at78339-78343MiB, even with0percent utilization. No GPU runs, service termination or queued-source changes occurred.

Native vLLM CPU prompt assembly/sampler/Msgpack/decoder diagnostic passed255exhaustive partitions of lengths1..8,15real-token patterns (en149/zh210/code384), and1reordered mixed batch with an unscheduled request. Synthetic logits and deterministic unfilled-memory canaries were used, with process-local PIN_MEMORY=False; CUDA remained uninitialized. This does not exercise GPU copies or Ray transport, and does not reproduce the real cold-v11 invalidID. Probev1 hit a CPU-only pinned-memory environment failure; v2 exposed an incorrect diagnostic rank oracle (installed batched_count_greater_than counts >= includingties); v3 uses the installed definition and passes. Evidence output/prompt-logprobs-debug/chunked-cpu-v3.{json,log}. No production algorithm change. Official Context7 and installedsource both show_sync_device before returning completedpromptlogprobs.

Added opt-in RL resume-training runner, offset-aware lifecyclevalidator and baseline-bound resumevalidator. Root integrationreview reproduced timestamped-checkpoint misbinding, empty/stale destination acceptance, missingoptimizer/stalescheduler acceptance, duplicate baseline source and malformedstep failures. Real eight-rank-layout tinyDCPfixtures exercise the existing structural inspector. Final local98passed2.48s. Native transfer/admission indices restart0in a freshprocess whiledriver/sentinel stepsremainabsolute; newvalidator preserves originalobservations and maps the ranges explicitly. Native-ordinalRED2failed54passed; earlier integrationRED9failed18passed. Independent review of this new toolwave remains pending; no GPU continuation accepted.

Ruling: prepare a fresh candidate resuming v9checkpoint19 to max_steps25 (5newupdates), retaining the existing10validnonzerogroup minimum and all8rank weight/reference/admission checks. This replaces the initial2-update preparation only; if rewarddiversity is insufficient it must fail rather than weakening the gate. Fullstateequality remains a separatecoldstate report; stochasticrollout bitwise continuation is not claimed. Candidate is not inserted into the existingGPUqueues and must await GPU and numerical prerequisites.


### 2026-09-16 05:14 UTC: native CPU acceptance passed; continuation supervisor waiting

Independent source snapshot /data_hdd/roll-qwen38-validation/sources/rl-resume-tools-v1/ROLL has1127files, source-manifestSHAca719c6ef14c1b0330a2d413ff65ad90b71cafc77ebb79433d86d0798d6d0953. Native PyTorch2.13CPU98passed4.65s. Actual typedRLVR preflight confirms baselineRUN/checkpoints/20260914-215136/checkpoint-19, savedstep19, restoredscheduler cursor160, firstupdate20,5remainingupdates andfreshHDDcheckpoint24 destination. Updatedvalidator reran historicalv9 successfully:20updates,21versions,58validnonzerogroups. Evidence archived in output/rl-validation-data/native-resume-tools-v1/{results.json,cpu-tests.log,preflight.log,v9-lifecycle.log}. No newGPUupdate occurred.

Added a separatepersistent supervisorPID4018908 at05:13:39UTC running environment/queue_rl_resume_v1.py (SHA427d81b07274dbfae61c675302b939e38c025080e16136a2160d03ef738528cf). Events environment/queued-rl-resume-v1/events.jsonl. It waits for promptv3exit0andshortparitypassed, backbonev3queue_completeandexactstatecomparison, then obtains the sharedGPUlock and waits for allGPUcompute processes torelease. It rerunspreflight, executes5resumedRLupdates andstrictresumeacceptance, failingwithoutGPUretries onanyprerequisite/training/validationfailure. Overallreleasegate remainsfalse. Originalv3/v8queues andsources unchanged.

05:14:39UTC livecheck: oldsupervisors3657791/3657792 andnew4018908 allalive. Oldevents stillwaiting_for_idle_gpus/waiting_for_prompt_diagnostic; neweventwaiting_for_prompt_and_backbone. All8GPUs remain78339-78343MiB and0percentutilization underexistingservice. Respectuserdecision: no service interruption.

Knownmissinggates remain: realpromptinvalidID/rootcause, fullROLLRaycoldlogprobcycle, natural2K/8Knumerics, fullbackboneSFT/RL, exportGPUload/fault/performance, independentfinalreview/vendor sync/FlashNextPR. The new queue is independent resumedLoRARL acceptance, not stochasticreplay or releasecompletion.


### 2026-09-16 06:38 UTC: GPU released; container-stop interruption and fresh v4 queues

The existing evaluation released GPUs and promptv3autostarted06:24:23UTC. It loaded8rankmodelweights but was interrupted by a Docker containerstop at06:26:04UTC (SIGTERM), thenSIGKILL10seconds later. Both ownvalidationcontainers were stopped; DockerOOMKilled=false. Promptv3exit137beforeanyprobabilityrequest; this is not an OverflowError reproduction. Backbonev3CPUpreflight failedbecausecontainerwasstopped; RLresumev1stoppedonpromptfailure. Preserve alloriginalfailureevents.

Root preservedstopstate/events in environment/container-stop-20260916.json, verifiedexactHDDcontaineridentity/sleepentrypoint andGPUidle, thenrestarted onlyroll-qwen38-hdd-validation. User subsequentlyconfirmedGPUavailabilityandcontinuation. Noothercontainerwasstoppedorrestarted.

Preparedfresh environment/prompt-id-v4 (6scripts,manifestSHA4a93f6074769465bbc23e6aecc2c79d8205042809510be563e0fcd08bdb758ac). Probeandworkerextensionarebyte-identicaltov3; newsupervisor/outputpaths only. Sourcecold-v11, finalexportedadapter, tolerancesandalltraining/v8/resume-tools-v1sourcesunchanged. At06:37:34UTCactivatedPID147084promptv4,147085backbonev4,147086RLresumev2. Events underenvironment/prompt-id-queue-v4,queued-backbone-v4,queued-rl-resume-v2. Promptoutputoutput/prompt-id-v4. Remainingbackbone/resumeoutputsunchangedbecauseGPUrunsneverstarted.

At06:38:17UTCpromptengineinitializing; backbonewaiting_for_prompt_diagnostic,RLwaiting_for_prompt_and_backbone. NoGPUtrainingstepclaim.


### 2026-09-16 07:38 UTC: ROLL bounded-loss fix and fresh v9 backbone run

The v4 native prompt diagnostic exited before generation: the installed runner is
`vllm.v1.worker.gpu.model_runner.GPUModelRunner`, whereas the observer targeted the
older `gpu_model_runner._get_prompt_logprobs_dict`. This is an instrumentation
failure, not a reproduction or resolution of the original prompt token overflow.

Backbone v8 started at 06:39:56 UTC and exited at 06:46:26 with CUDA OOM in initial
held-out validation, before any optimizer update. ROLL's global MTP `_postprocess`
patch materialized full vocabulary logits and bypassed the Qwen chunked-loss hook.
The failure requested another 3.79 GiB while only about 3.7 GiB was free.

Added a Qwen-only labeled postprocess branch that calls its bounded vocabulary
loss in training and eval, preserving tied/untied projection weights and keeping
unsupported inference-context handling explicit. Added actual ROLL MTP-patched
GPU model regression for train/eval, TP1/TP2, and tied/untied weights. RED2 observed
320 projected rows despite chunk size 256. Both `.181` ranks then passed the full
20-test integration suite (155 seconds on the reported rank). Evidence:
`output/cold-restore-memory/backbone-v9-{red2,green}.log`. Existing v8 sources remain
unchanged. The first test-observer attempt was invalid because a global dispatch
mode interfered with FlexAttention; RED2 scopes observation to postprocess.

Frozen source `/data_hdd/roll-qwen38-validation/sources/sft-backbone-v9/ROLL` has
1123 manifest entries, SHA256
`971295c65d05d64422968508d5842af073a6c8be67c649d56ca209092e11d22e`.
Supervisor PID 341035 uses `environment/queue_backbone_v5.py`, with events/logs in
`environment/queued-backbone-v5`. CPU preflight passed and the full run started
07:31:23 UTC. At 07:38:18 UTC all 64 initial validation batches had completed in
101.93 seconds; no successful optimizer update was yet confirmed. It retains the
100-step / checkpoint-50 continuation / complete state-comparison acceptance.
HDD outputs are `sft-backbone-100step-8k-v9`, `sft-backbone-100step-8k-resume-v9`, and
`backbone-continuation-exact-v9.json`.

Prompt v5 instruments native V2 target-ID, top-k, worker-result, async CPU-result,
and frontend decode boundaries without changing arithmetic, values, or error
propagation. Tensor observations synchronize CUDA and are diagnostic only.
Actual V2 class installation and a five-token native GPU Triton fixture passed;
this is not full-model probability parity. Files live in
`output/prompt-logprobs-debug/v5`, remote `environment/prompt-id-v5`.

New queued supervisors at 07:35:40 UTC: PID 381478 waits for the entire backbone
queue to terminate before running prompt v5; PID 381479 launches the existing
five-update RL continuation only after prompt v5 parity AND backbone v9 exact
state equality pass. Shared GPU lock and idle checks preserve other workloads.
Queue script manifest SHA256
`25f30fd8f2ec2b7585f4cd69daf71392b9981ea2f3b8e2df6655805d5949450c`.
No failed one-shot queue was restarted. Overall release remains incomplete.


### 2026-09-16 08:25 UTC: first-gradient NCCL memory fix; v10 running

Backbone v9 completed all 64 initial validation batches, then failed the first
backward gradient reduce-scatter (exit 1 at 07:41:04 UTC). NCCL could not allocate
128 MiB outside PyTorch's allocator. Rank2 recorded 73,353,307,136 bytes allocated,
79,010,201,600 reserved, peak 78,229,739,008; update_successful=false. No successful
backbone update is established by v9. Preserved log:
`output/cold-restore-memory/backbone-v9-nccl-oom.log`.

A two-rank standalone NCCL experiment reproduced OOM with almost all GPU memory
held in unused PyTorch cache; freeing the cache before first reduce-scatter made
both ranks pass and return exact expected values. RED/GREEN logs are
`output/cold-restore-memory/nccl-cache-{red,green}.log`. Added
`FirstGradSyncCacheRelease` and gated its use to CPU-offloaded optimizers with
synchronous gradient reduction. It releases unused cache before the first
successful grad-finalization call, without altering tensors, return values or
exception propagation. Two local helper regressions passed; the actual production
wrapper also passed the two-rank GPU pressure test (`nccl-cache-production-green.log`).
Independent scoped reviews found no blocking defects in either the bounded-loss
integration or the cache-release integration; reports are in the SDD directory.
This does not establish full-model completion or overlapping-gradient support.

Frozen v10 source: `/data_hdd/roll-qwen38-validation/sources/sft-backbone-v10/ROLL`,
1127 manifest entries, SHA256
`7cd4448946c68d83bb6b14b2b065e5ef6d80f932613b806b748115d3993746a2`.
Supervisor PID535887 uses `environment/queue_backbone_v6.py` and
`environment/queued-backbone-v6/{events.jsonl,backbone-100step.log}`.
Preflight passed; v10 started 08:25:00 UTC. Output suffixes are v10, and the queue
retains full 100 steps, checkpoint50 -> 49-update continuation, complete comparison.
No active/old snapshot or shared dependency was modified.

Prompt v5 ran after v9 exited and finished all 9 requests (base, exported final
adapter, post-sleep adapter; en/zh/code). No invalid token IDs were observed.
Native worker, async CPU, frontend and selected-token outputs agree exactly;
post-sleep probabilities are bitwise identical. The instrumented run synchronizes
CUDA, so the original uninstrumented overflow remains unresolved. Adapter-vs-actor
short parity FAILED: en max_abs4.618779/mean_abs0.494292, zh0.853354/0.098915,
code2.329251/0.219183, unchanged atol/rtol0.02. Prompt exit1 was a numerical mismatch,
not an observer/API failure. New RL queue v3 stopped on failed prerequisites;
no RL updates were launched.

Local analysis corrected provenance: frozen cold-v11 actor config is TP1/EP8,
not TP2, while native vLLM uses TP8/EP8. The actor's eight rows are repeated input
samples returned by DP_MP_DISPATCH_FIRST, not eight TP-rank dumps. The384-token code
case contains no answer token (its prompt alone is394tokens). Existing short-probe
artifacts therefore must not be described as answer-token or rankwise acceptance.
See `output/prompt-logprobs-debug/v5/parity-analysis.md`. Next numerical control is
restored-actor adapter-enabled/disabled on identical inputs; no export defect or
scale correction has been established. RL tool tests reran locally:98passed2.57s.


### 2026-09-16: v10 first backbone update; TP4 candidate

Archived the v10 queue and all eight rank metrics in
`output/cold-restore-memory/backbone-v10-evidence.tar.gz`; the derived update
receipt is `backbone-v10-update-summary.json`. Every rank reports step0
`update_successful=true`; rank6 reports step1=false. The run ended at08:35:39UTC
on a632MiB MoE permutation allocation with617MiB free. It did not complete
100updates or produce continuation acceptance. This is the first successful
full-text-backbone update, with the N-gram table frozen, not a complete SFT pass.

TP4/EP8/ETP1 with accumulation2 preserves global batch4 while reducing dense
parameter/gradient replication per GPU. The existing v11 candidate is not yet
frozen. Its original eight-rank regression exposed a QSA output gate left
unsharded when KV groups are fewer than TP ranks. A direct global-projection
regression also failed on all ranks: gate heads8 versus local query heads2.
`backbone-v11-tp4-gate-red.log` preserves that failure. The QSA override now uses
the same within-replicated-group TP rank to slice the gate, and preserves gates
already sliced by newer Megatron versions. Eight-rank value/gradient and full
model loss regressions are running; no TP4 training acceptance is claimed.

A separate actor-only paired adapter diagnostic is being prepared locally for
enabled-before -> disabled -> enabled-after on the same checkpoint19 and token
inputs. It must preserve zero updates and no checkpoint writes. Prompt-v5 parity
remains failed, and all final release gates remain open.


### 2026-09-16 10:27 UTC: TP4 distributed regressions and preflight passed

The actual eight-rank TP4/EP8 run exited0:9tests passed on each rank. Gate values
and QKV gradients match an independently assembled global projection exactly for
KV group counts1/2/4; ROLL-patched bounded loss and full DDP gradient comparisons
also pass (tied/untied, train/eval). Evidence:
`output/cold-restore-memory/backbone-v11-tp4-green.log` and
`backbone-v11-regression-receipt.json`. These are tiny-model GPU regressions,
not full125.74B training acceptance.

Frozen v11 has1129source hashes; manifestSHA256
`9af5dc75156db2b751e993c939084254a826bb06ab1e44016d958a40591130ca`.
CPU preflight verifies all hashes, TP4/DP2/EP8/ETP1 and frozen external N-gram
scope. The production candidate SFT config now usesTP4 withaccumulation2,
preserving globalbatch4. Full100steps/checkpoint50 continuation remain pending.

Actor paired diagnostic has8local CPU harness passes. First real CPU preflight
failed because BaseConfig replaces configured GPU counts with visible counts
while CUDA is hidden; the diagnostic now keeps requested topology checks in CPU
preflight and requires actual8GPU/one-node discovery for execution. The first
source/output remain preserved; a new actor-pair-v2 source is copied from all1114
verified cold-v11 files plus the isolated probe. No optimizer updates or model
acceptance are implied by these harness/preflight checks.


### 2026-09-16 10:49 UTC: actor diagnostic running, v11 queued

Actor source `/data_hdd/roll-qwen38-validation/sources/actor-pair-v3/ROLL` is
verified against1114cold-v11 hashes plus the isolated diagnostic. The original
checkpoint19 and old sources remain unchanged. Native CPU preflight passed;
local diagnostic controls now9passed after preserving nonfinite historical
tracker fields explicitly in valid JSON. Newly measured probabilities still
require finite values, and before/after adapter probabilities require exact equality.

Persistent actor supervisorPID1008638 uses `environment/actor-pair-v3/queue_remote.py`
with events/logs in`environment/queued-actor-pair-v1`; output is
`output/actor-pair-v1-20260916-run03`. Backbone supervisorPID1008639 uses
`environment/actor-pair-v3/queue_backbone_v7.py` and waits for the diagnostic to
return, then for idle GPUs under the shared lock. Its v11 sequence retains
100updates, checkpoint50 ->49updates, and full checkpoint state comparison.
The actor diagnostic result is not a backbone or release acceptance gate; all
previous numerical failure evidence is preserved. No new successful update
from v11 is yet established.


### 2026-09-16 11:01 UTC: backbone initial validation passed; actor retry queued

Backbone v11 began at10:53:21UTC and completed64initial validation batches in
178.2232s. It is now executing the first training update; no successful v11
update is yet established. Live GPU usage is about72,896–73,532MiB.

Actor run03 returned1at10:53:06UTC before restore/probability collection: its
mutation guard read `ChainedOptimizer.optimizer`, which asserts for multiple
children. This was a diagnostic-helper error, not a measured model mismatch.
The guard now traverses `chained_optimizers` without reading that property.
Native CPU check on the installed Megatron blocks root plus both children and
keeps CUDA uninitialized; local control tests9passed. Frozen v4 diagnostic
source uses the same1114verified cold-v11 source files plus the fixed probe.
Its CPU preflight passed11:00:52UTC. SupervisorPID1099042 waits for the entire
backbone-v7 queue to return, then obtains the shared GPU lock. Events are
`environment/queued-actor-pair-v2/events.jsonl`; source/launcher are
`sources/actor-pair-v4/ROLL` and`environment/actor-pair-v4/run_remote.sh`; output
is`output/actor-pair-v1-20260916-run04`. No existing source/run was overwritten.

QSA activation arithmetic was also checked against the pinned HF source and
native vLLM source: both multiply by BF16 gate.sigmoid. The TP4 fix deliberately
preserves that operation; generic Megatron's separate FP32 gate implementation
is not evidence for changing this model's arithmetic.


### 2026-09-16 11:19 UTC: five globally completed TP4 updates

Live check confirms all8ranks completed steps0–4 successfully; step5 is in
progress (four ranks already reported success). No rank has recorded a failed
update. Cumulative PyTorch CUDA peak is72,576,902,656B per rank(67.5925GiB).
Last globally completed step4 took159.181s; loss1.6293753833. All inputs have
shape[2,8192] per worker, but step4's maximum valid sequence is595tokens.
Padded8K execution is not evidence of completion on full8K valid text.
The100-update, checkpoint50 continuation and complete comparison gates remain open.
Receipt:`output/cold-restore-memory/live-20260916-111901.json`.

Supervisor1008639 continues the full backbone-v7 queue; supervisor1099042 waits
for that entire queue to finish before running the repaired actor-pair-v4
diagnostic under the shared lock. No actor-pair probabilities have yet been
measured. The source of the previous diagnostic failure and the native CPU guard
regression are archived under`output/prompt-logprobs-debug/actor-pair-v1/`.

A fresh static full-backbone RL capacity follow-up uses the current eight-rank
parameter census:270.4568GB BF16 role storage, versus408.8816GB observed host
headroom. Actor+reference private CPU copies do not fit that SFT headroom; this
is an estimate, not a measured RL OOM. See
`output/cold-restore-memory/backbone-rl-capacity-followup.md`. Sleeplevel2 also
requires explicitly proving immutable native N-gram table retention/restoration.
No RL memory policy was changed, and full-backbone RL is not launch-ready.
Numerical parity, long-text/full SFT, real RL continuation, backbone RL, export,
fault/performance, vendor synchronization and Flash-Next PR remain incomplete.

### 2026-09-18 GDN convolution and update-gate precision

Native/actor recurrence captures reproduced independently on .181. First-token algebra identifies an extra BF16 rounding boundary before native SiLU; FLA fused convolution omits it. The Qwen4 override now uses FLA convolution with activation=None followed by the existing SiLU, with an explicit Megatron hook preserving other models' default fused behavior. Beta is promoted before sigmoid. Checkpoint keys, parameter storage, routing and tolerances are unchanged. Both regressions failed on the original real GDN; the isolated patched ROLL/Megatron snapshot passed 17 tests (including four CPU norm tests). Eight-GPU actor parity is running; this does not establish complete numerical or SFT/RL/OPD acceptance. Apply the new SHA-locked gate and convolution patches in the manifest order; do not patch running shared dependencies.

### 2026-09-18：检查点填充修复与训练验证解耦

- 完整 SFT v14 比较读完 285140 个存储项、8 个 worker RNG；发现 864 个差异。
- 原地只读归因确认 864 个差异全部是 12 元素参数后的 52 元素对齐填充，均结束于 64 元素边界；无其他失败。原始载荷 exact 仍为 false，报告与载荷不改写。
- 已定位 Megatron dp_reshardable 使用 torch.empty 写出填充；新增 SHA 锁补丁改为 torch.zeros。实际依赖合成 CPU 回归 RED 2 failed / GREEN 2 passed；原 comparator 与 integrity 本地 28 passed，比较器未改。
- 以最新 GR/GDN 和确定性填充依赖开展独立 RL20 + 冷恢复更新验证；数值、SFT 原始载荷一致性和最终发布仍分别保留，不以独立生命周期冒充整体通过。
