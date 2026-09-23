# AI Numerical Programming for Experienced C and DSP Developers

## Connecting low-level numerical practice to modern AI validation, frameworks, and CPU/GPU backends

## Reader profile

This guide is for developers who are already comfortable with C or C++, DSP, embedded systems, Win32 or native APIs, fixed-size buffers, hand-written numerical kernels, memory layout, and low-level debugging.

The missing bridge is not programming ability, and the reader does not need to become a Python specialist. The bridge is the set of conventions and tools that modern AI work places around numerical code:

- tensor shape, stride, dtype, device, and aliasing contracts;
- automatic differentiation and gradient validation;
- Python and notebooks as orchestration and reference tools;
- framework operator registration and model exchange;
- CPU libraries, GPU execution, mixed precision, and quantization;
- reproducible build, test, benchmark, and artifact workflows.

## 0. Purpose

This is not an introduction to C or C++ syntax, and it is not an argument that low-level code should be replaced by a Python framework.

It is a practical guide for preserving low-level numerical expertise while connecting it to contemporary AI implementation and validation.

The central workflow is:

```text
state the numerical and memory contract
→ build a tiny transparent reference
→ compare against an independent executable oracle
→ verify backward and state updates when training is involved
→ integrate through an explicit operator boundary
→ profile the real bottleneck
→ move to optimized CPU or GPU backends one layer at a time
```

RNNs and LSTMs appear in this guide, but only as one stateful case study. The main subject is broader: dense operators, reductions, losses, normalization, quantization, custom operators, CPU/GPU kernels, and the engineering system that makes their results trustworthy.

---

## Choose the shortest useful route

| Current situation | Start here | First discriminating action |
|---|---|---|
| The modern AI stack feels opaque even though the arithmetic is familiar | Sections 1–5 | Identify which unfamiliar layer owns the behavior before changing the kernel |
| You need a place to practice without buying a new workstation | Section 6 | Start on local CPU, then use a hosted GPU only for the question that requires it |
| You want one end-to-end learning path instead of a tool catalogue | Sections 5.4 and 6.4–6.7 | Carry one dense/loss fixture through C, NumPy, PyTorch, controlled CPU timing, and optional GPU timing |
| You have a C/DSP kernel and want to use it from modern AI code | Sections 1, 3, 20, and 21 | Write the tensor and mutation contract before binding the pointer |
| A framework result and native result disagree | Sections 7–11 | Freeze one tiny input and compare the first divergent tensor |
| Forward output agrees but training fails | Sections 12–14 | Check analytic gradients, reductions, and optimizer state separately |
| CPU scalar code is correct but a library backend disagrees | Sections 10, 23, and 26 | Record exact shapes, strides, transpose flags, and reduction policy |
| GPU code is slower than CPU code | Sections 24 and 26 | Separate transfer, launch, synchronization, and steady-state kernel time |
| FP16/BF16/TF32 changes the result | Sections 14 and 17–19 | Identify storage, compute, and accumulation precision for every reduction |
| INT8 or fixed-point inference loses accuracy | Section 18 | Inspect scale, zero point, calibration range, saturation, and accumulator width |
| Full-sequence attention works but cached decoding does not | Sections 10.9 and 13.4 | Compare absolute query/key positions, mask polarity, cache update timing, and chunk/full output |
| A notebook works but the native build is fragile | Sections 20 and 25 | Move contracts and fixtures into files and CTest; keep the notebook as a client |
| A sanitizer-clean program still gives plausible but wrong output | Sections 8–11 | Use non-square diagnostic fixtures and independent numerical comparison |
| You need a defensible performance claim | Sections 26 and 28 | Pass the correctness gate, then report timing scope, shapes, hardware, and variance |

## Guide map

- **Part I, Sections 1–6:** translate C/DSP experience into tensors, frameworks, and a practical learning environment.
- **Part II, Sections 7–16:** define contracts and verify operators, gradients, state, training steps, and model/task acceptance.
- **Part III, Sections 17–19:** control numerical stability, precision, quantization, reproducibility, and checkpoints.
- **Part IV, Sections 20–24:** connect native code to Python, PyTorch, ONNX Runtime, CPU libraries, and GPU execution.
- **Part V, Sections 25–28:** build, test, benchmark, diagnose, and state defensible completion claims.

---

# Part I — Translate existing skills into the AI stack

## 1. What carries over from C and DSP

Much of the required intuition is already familiar. The vocabulary changes more than the underlying engineering.

| C/DSP practice | AI-system counterpart | Important difference |
|---|---|---|
| Array plus explicit length | Tensor plus shape and strides | Tensor views may be non-contiguous and may alias |
| FIR/IIR state | Recurrent, cache, or running-statistics state | State may also participate in automatic differentiation |
| Block processing | Batch, chunk, tile, or sequence processing | The reduction and update denominator must be explicit |
| Test vector and expected waveform | Golden tensor fixture | Tolerance may depend on dtype, scale, backend, and reduction order |
| MAC loop | Dot product, GEMM, convolution, attention | Backends may reorder, fuse, tile, or lower the same operator differently |
| Q-format and saturation | Quantization scale, zero point, clipping, calibration | Parameters and activations may use different schemes and granularities |
| SIMD intrinsic | Vector ISA, tensor instruction, warp-level primitive | The fastest shape and layout may be hardware-specific |
| DMA and double buffering | Host/device transfer, pinned memory, streams | GPU execution is normally asynchronous |
| Oscilloscope probe point | Tensor dump, hook, trace, profiler range | The earliest divergence is usually more useful than final loss |
| Reference C implementation | Eager framework or double-precision oracle | A framework is strong evidence, but not infallible |
| Cross-platform HAL | Operator schema plus backend dispatch | Mutation, aliasing, dtype, device, and shape inference are part of the API |

The useful mental shift is:

> A tensor is not merely a pointer to numbers. It is a numerical object with shape, strides, dtype, device, ownership, aliasing, mutation, and often gradient semantics.

## Transition map: what will feel unfamiliar

Reports from experienced native-code developers entering Python/ML environments show a repeated pattern: the arithmetic is usually not the hardest part. The confusion comes from invisible dispatch, state, metadata, and environment boundaries. Treat these as scope signals, not as universal claims about every developer.

| First impression | More accurate model | First useful check |
|---|---|---|
| “Python is executing the numerical loop.” | Python often describes bulk operations that dispatch into native CPU or GPU kernels. | Compare a Python element loop with one NumPy/PyTorch operator, then profile below the Python call. |
| “Vectorization means SIMD.” | In this ecosystem, *vectorized* may mean array programming, batching, compiler SIMD, or GPU parallelism. | Name the level explicitly: expression, batch, ISA, or kernel. |
| “A tensor is a multidimensional C array.” | A tensor is a view over storage plus shape, strides, dtype, device, layout, ownership, and possibly graph history. | Print all metadata and test a transposed or sliced non-contiguous view. |
| “Running the same notebook means running the same program.” | A notebook is a stateful interactive session; cell order and stale variables can affect results. | Restart the kernel, execute from top to bottom, and compare with a clean command-line run. |
| “The source tree defines the build.” | The interpreter, installed wheels, native libraries, driver, toolkit, framework build, and environment variables are also part of the executable system. | Record the selected interpreter and package/backend versions before debugging code. |
| “`backward()` computes a derivative and returns it.” | Autograd records a dynamic graph, saves values, routes gradients, and usually accumulates into `.grad`. | Inspect leaf/non-leaf status, saved state, mutation, and whether gradients were cleared. |
| “The GPU call finished when the host call returned.” | GPU launch and transfer APIs are commonly asynchronous. | Synchronize only around the measurement boundary and separate transfer from kernel time. |
| “The model is the C function.” | A deployable model may include weights, operator schemas, preprocessing, state, dtype/layout assumptions, and runtime configuration. | Inventory artifacts and contracts, not only source functions. |
| “If values match, integration is correct.” | Framework integration also requires correct output metadata, aliasing, mutation, device dispatch, backward behavior, and export behavior. | Test metadata and composition, not only element values. |

These transitions suggest the guide's scope. It should not teach all of Python, data science, or model theory. It should explain the boundaries that hide numerical behavior from an otherwise experienced systems programmer.

### Vocabulary collisions

Several common words carry different meanings across native, DSP, and AI contexts. Clarify them before diagnosing a problem.

| Word | Possible meanings | Safer wording |
|---|---|---|
| vectorization | array expression, batch processing, compiler SIMD, explicit intrinsics | state the exact level |
| kernel | OS kernel, numerical routine, CUDA device function, framework operator implementation | use *CPU routine*, *CUDA kernel*, or *operator kernel* |
| graph | dataflow/autograd graph, exported IR, model topology | name eager autograd, compiled graph, or exchange graph |
| state | algorithmic state, optimizer state, RNG state, notebook process state, runtime cache | name owner and lifetime |
| batch | independent examples, temporal block, microbatch, hardware tile | state semantic axis and reduction rule |
| model | equations, module object, parameter file, exported graph, complete preprocessing-to-output system | state the artifact boundary |
| parameter | trainable tensor, function argument, configuration setting | say trainable parameter when gradients apply |
| package | Python import package, installable distribution, native library package | identify the installer and artifact type |
| runtime | language runtime, framework dispatcher, inference engine, GPU runtime | identify the specific layer |

## 2. A modern AI numerical program is a stack

Do not diagnose every failure at the kernel level. Separate the layers.

### 2.1 Model and mathematical layer

Defines:

- equations and operator ordering;
- loss and metric definitions;
- training versus inference behavior;
- state, randomness, and normalization semantics.

### 2.2 Tensor and operator layer

Defines:

- shape and broadcasting;
- dtype and device;
- strides and layout;
- mutation and aliasing;
- output metadata;
- backward behavior.

### 2.3 Graph and differentiation layer

Defines:

- how operations compose;
- which values must be saved for backward;
- gradient accumulation;
- graph breaks or opaque custom operators;
- tracing, compilation, export, and shape propagation.

### 2.4 Kernel and backend layer

Defines:

- scalar, SIMD, BLAS, oneDNN, CUDA, or accelerator implementation;
- work partitioning and reductions;
- temporary workspace;
- precision and algorithm choice;
- backend-specific layout.

### 2.5 Runtime and orchestration layer

Defines:

- allocation and memory pools;
- host/device transfer;
- streams, threads, synchronization, and scheduling;
- data loading;
- checkpointing and logging.

### 2.6 Experiment and evidence layer

Defines:

- exact source revision and build;
- fixtures and expected outputs;
- environment and random seeds;
- comparison tolerance;
- benchmark scope;
- artifacts that support a public claim.

A plausible final output can coexist with an error in any lower layer. Diagnose the earliest layer that can explain the observation.

## 3. Three useful working modes

### 3.1 Reference mode

Goal: transparency and correctness evidence.

Typical choices:

- very small shapes;
- deterministic diagnostic values;
- scalar loops;
- `double` for reference calculations;
- no fast-math;
- complete intermediate dumps;
- single thread;
- explicit synchronization.

### 3.2 Integration mode

Goal: make a verified numerical operation participate correctly in a framework or runtime.

Typical concerns:

- operator schema;
- shape and dtype validation;
- mutation and aliasing declaration;
- CPU/GPU dispatch;
- autograd or inference-only behavior;
- serialization and export;
- build and ABI boundaries.

### 3.3 Performance mode

Goal: reduce latency, increase throughput, or reduce memory without losing the established contract.

Typical choices:

- optimized layout;
- vectorization or vendor library;
- mixed precision or quantization;
- fusion;
- asynchronous execution;
- fewer diagnostic checks in the hot path.

Do not use performance mode as the only implementation. Keep a small reference path and the fixtures that connect the optimized path back to it.

## A minimum working bridge

Do not begin by installing every framework or learning the whole Python ecosystem. Build the bridge in layers.

1. **Shell and isolated Python environment** — know which interpreter is running; create one project-local virtual environment; record dependencies.
2. **NumPy array semantics** — learn shape, dtype, broadcasting, views, strides, reductions, and the difference between a Python loop and a bulk native operation.
3. **PyTorch on CPU** — learn tensor metadata, autograd, gradient accumulation, modules, optimizer state, and eager reference code without adding GPU complexity.
4. **Notebook as a microscope, script/test as authority** — use notebooks for inspection and plots, but require restart-and-run-all plus a clean non-interactive test.
5. **Native boundary** — call one verified C/C++ operation through a stable ABI, binding, or registered custom operator; test ownership and non-contiguous inputs.
6. **GPU only after the CPU contract is stable** — then add device placement, transfer, synchronization, profiler timelines, and backend-specific tolerances.
7. **Export/deployment last** — treat ONNX or another runtime as a new implementation boundary with its own tests.

A later worked example follows one small computation through these layers. The public guide remains self-contained; executable fixtures used to validate the material are editorial evidence rather than required companion downloads.

This order is deliberately conservative. It prevents Python packaging, autograd, framework registration, and GPU timing from becoming one undifferentiated problem.



## 4. Why familiar C can be slower than expected

The old skills are still valuable, but the machine and software around the loop have changed. A hand-written loop may lose for reasons that have little to do with the number of arithmetic instructions in its source.

The useful question is not:

> Why is C slow?

It is:

> Which layer prevents this particular C implementation from reaching the useful hardware, and which replacement has the lowest total cost?

### 4.1 Eliminate false comparisons first

Before changing code, rule out measurement mistakes.

| Apparent problem | Discriminating check |
|---|---|
| A framework operation is much faster | Verify that both paths perform the same operation, dtype conversion, reduction, and output materialization |
| Debug C is slow | Compare a strict optimized build, not only a debugger build |
| GPU timing looks impossibly fast | Synchronize or use device events; separate launch from completion |
| The first call is slow | Separate compilation, allocation, cache warm-up, and library initialization from steady state |
| The C loop is fast in isolation but the application is slow | Measure copies, layout conversion, allocation, thread startup, and dispatch around it |
| More threads are slower | Record every active thread pool and test one controlled parallelism budget |
| A benchmark result changes after logging is removed | Check timing scope, dead-code elimination, thermal state, and synchronization |

A modern framework may also cache algorithms, compile graphs, reuse memory, fuse operators, or call a highly tuned vendor library. Compare complete contracts and complete timing scopes, not source-line counts.

### 4.2 The CPU is not a faster version of the old scalar machine

Current CPUs combine:

- out-of-order execution;
- speculative execution and dynamic branch prediction;
- multiple vector instruction sets;
- deep cache hierarchies and hardware prefetchers;
- simultaneous multithreading;
- many physical cores;
- shared last-level cache and memory controllers;
- sometimes heterogeneous performance and efficiency cores;
- sometimes several NUMA domains.

A loop can therefore be limited by very different resources:

```text
front-end / instruction delivery
execution ports
loop-carried dependency
branch misprediction
L1 or TLB behavior
last-level cache misses
memory bandwidth
synchronization or coherence traffic
```

Do not optimize all of these at once. Use a profiler and a deliberately chosen shape matrix to identify the active limit.

### 4.3 Why the compiler may not vectorize an obvious loop

Common blockers include:

- possible pointer aliasing;
- loop-carried dependencies;
- unknown trip count or awkward remainder handling;
- calls that cannot be inlined or analyzed;
- mixed types and implicit conversions;
- reductions whose reassociation would change floating-point behavior;
- control flow that is expensive to predicate;
- unknown alignment;
- data structures that force gathers or non-unit strides.

The first response should be a vectorization report and assembly inspection, not immediate intrinsic code.

If the contract truly guarantees non-aliasing, alignment, or a fixed multiple of the vector width, express that guarantee in a compiler-supported and testable way. Do not add `restrict`, alignment assumptions, or fast-math merely to silence a report; a false promise is undefined behavior or a numerical contract change.

### 4.4 Memory layout often matters more than arithmetic

Many AI operators reuse a small number of arithmetic patterns. Performance often depends on whether data arrives in the order expected by the backend.

Typical penalties include:

- array-of-structures where structure-of-arrays would vectorize better;
- inner loops over a non-unit stride;
- repeated transpose or packing;
- temporary tensors larger than cache;
- materializing broadcast values unnecessarily;
- conversion between framework-preferred and native-preferred layouts;
- copying across an ABI because ownership or contiguity was not declared.

Count bytes as well as operations. For a bandwidth-bound operator, removing one full read/write pass can matter more than replacing several scalar instructions.

### 4.5 Branches are a measurement question, not a superstition

Modern predictors handle many regular branches well. Branchless code can still be faster for unpredictable data, but it may also:

- execute work that would otherwise be skipped;
- increase instruction count and register pressure;
- force extra loads;
- prevent short-circuit behavior;
- create a less useful vectorization pattern.

Test representative data distributions. Report branch misses, cycles per element, and memory behavior before claiming that the branch itself was the bottleneck.

### 4.6 Multicore scaling is a separate program

Turning one loop into several threads introduces:

- partitioning and scheduling cost;
- synchronization and barriers;
- reduction and merge work;
- cache-coherence traffic;
- false sharing;
- memory-bandwidth contention;
- NUMA placement;
- nested library or framework thread pools;
- different floating-point reduction order.

The program can reach the memory-bandwidth limit with only a subset of cores. It can also become slower because an outer task pool invokes an internally threaded BLAS or framework operator.

Treat the total thread budget as an explicit resource:

```text
application workers
× framework inter-op work
× framework intra-op work
× BLAS/OpenMP threads
× data-loader/process workers
```

The product is not always the literal number of runnable threads, but it is a useful oversubscription alarm.

### 4.7 Standard operators have an enormous tuning advantage

A plain C implementation of GEMM, convolution, normalization, or attention competes with systems that may use:

- architecture-specific microkernels;
- cache blocking and packing;
- ISA dispatch;
- post-op fusion;
- autotuned algorithms;
- graph-level fusion;
- specialized matrix or tensor instructions;
- persistent memory pools;
- shape-specialized compilation.

The right move is usually to map standard work to a library first. Hand-written code remains valuable as:

- the transparent reference;
- the implementation of unusual semantics;
- the small fixed-shape or hard-real-time path;
- the preprocessing and postprocessing around a library call;
- the fallback for unsupported hardware;
- the measured custom kernel after standard paths have been exhausted.

### 4.8 GPU acceleration has a crossover point

A GPU can lose when:

- the workload is too small;
- host/device transfer dominates;
- each launch does too little work;
- synchronization is inserted between every operation;
- layout conversion or allocation dominates;
- the CPU has a strong vectorized library path;
- the GPU kernel has poor occupancy or memory access;
- only one isolated operator is moved while surrounding work stays on the CPU.

The useful unit of migration is often a pipeline or fused region, not one scalar-looking loop.

### 4.9 A practical diagnosis ladder

Use this order before rewriting the kernel:

```text
1. confirm identical semantics and timing scope
2. use an optimized strict build
3. inspect vectorization and generated code
4. measure cache, bandwidth, branches, and cycles
5. measure single-thread behavior before multicore behavior
6. cap and record every thread pool
7. compare a standard library/backend mapping
8. include layout conversion and dispatch costs
9. test graph/compiler fusion when operations form a chain
10. test GPU only at shapes and pipeline scopes that can amortize overhead
```

This ladder preserves the main advantage of experienced C work: the ability to isolate and measure a system instead of guessing from abstractions.

## 5. The highest-leverage modernization path

The target is not a conversion from C programmer to Python programmer. It is a hybrid numerical developer who can use Python and frameworks to control experiments while retaining ownership of contracts, native kernels, and hardware behavior.

### 5.1 Learn these concepts first

| New skill | Why it has high leverage | Minimum useful competence |
|---|---|---|
| NumPy array semantics | Provides a compact independent oracle and teaches views, broadcasting, reductions, and dtype behavior | Express a fixture without Python element loops; inspect shape, strides, dtype, and ownership |
| PyTorch eager tensors and autograd | Makes gradients and state transitions executable and inspectable | Reproduce one native operator, call backward, inspect accumulated gradients, and reset state correctly |
| Python environment discipline | Prevents invisible interpreter/package drift | Create one project environment, print the active interpreter and package versions, reproduce it from files |
| Notebook discipline | Makes interactive work useful without treating hidden state as authority | Restart-and-run-all, export decisive code to a script/test, preserve fixtures outside the session |
| Profiling boundaries | Prevents optimization of the wrong layer | Separate Python overhead, native operator time, copies, compilation, launch, synchronization, and steady state |
| Framework operator contracts | Lets existing C/C++ participate in autograd, compilation, vectorization transforms, and export | Declare schema, mutation, aliasing, metadata behavior, device dispatch, and backward support |
| CPU parallelism budget | Avoids oversubscription and misleading scaling | Identify inter-op, intra-op, BLAS/OpenMP, process, and application worker counts |
| GPU execution semantics | Prevents stale results and false timing | Understand device memory, asynchronous launch, streams/events, transfer, and error observation |
| Numerical validation | Distinguishes approximation from implementation error | Use stable references, finite differences, independent backends, and scale-aware tolerance |

### 5.2 Defer these until the problem demands them

Do not front-load the entire modern ecosystem.

Usually defer:

- advanced Python language features unrelated to experiment control;
- distributed training;
- orchestration platforms and cluster administration;
- custom graph compiler passes;
- writing a GEMM or convolution kernel from first principles;
- hand-written CUDA/HIP before a library mapping is measured;
- full model training before operator fixtures are trustworthy;
- every packaging system at once;
- every accelerator vendor at once.

These are valuable branches, not entry requirements.

### 5.3 Choose the lowest layer that solves the measured problem

| Situation | First implementation choice | Escalate when |
|---|---|---|
| Tiny fixed shape, embedded target, hard latency bound | Readable C/C++ with explicit memory and fixed tests | Compiler output or measured latency misses the target |
| Regular CPU loop | Optimized compiler build and vectorization report | The compiler cannot express the needed layout/algorithm or misses a measured target |
| Standard dense/conv/reduction operation | BLAS, oneDNN, or framework-native operator | Conversion overhead dominates or semantics are not representable |
| Several framework operators are bandwidth/dispatch bound | Framework compiler or existing fused operator | Graph breaks, unsupported semantics, or generated code remains the bottleneck |
| Existing native routine with unusual semantics | Registered custom operator around the verified routine | Backend-specific implementation is required |
| Standard large GPU operation | Vendor library or framework-native GPU operator | A measured shape/semantic gap remains |
| Custom GPU elementwise/reduction/fusion | Triton or another kernel DSL where supported | Lower-level control, portability, or unsupported behavior requires CUDA/HIP/SYCL |
| Architecture-specific GPU kernel | CUDA, HIP, or vendor-specific low-level path | Portability or maintenance cost outweighs the benefit |
| Portable deployment graph | ONNX plus a tested runtime/backend | Exported semantics or a required operator cannot be represented |

This table is a starting point, not a hierarchy of prestige. The lowest layer is not automatically faster, and the highest layer is not automatically portable.

### 5.4 Modernize one kernel end to end

Use one small operation as the learning vehicle.

```text
A. scalar double reference with diagnostic values
B. strict optimized C/C++ implementation
C. compiler-vectorization report and single-thread profile
D. NumPy oracle using the same fixture
E. PyTorch CPU forward and, when relevant, autograd
F. standard library/backend mapping
G. native custom-operator boundary if the kernel is still needed
H. compiled/framework path and graph-break inspection
I. GPU library path, then custom GPU kernel only if justified
J. one comparison report covering correctness, layout cost, timing scope, and environment
```

Do not change the fixture at each step. A stable fixture is the thread connecting all implementations.

### 5.5 Framework compilers are another implementation boundary

A compiler such as `torch.compile` can fuse operations and reduce dispatch overhead, but it does not make arbitrary Python free. Graph breaks, data-dependent behavior, unsupported operators, recompilation, dynamic shapes, and custom-call boundaries affect the result.

Use framework compilation after the eager reference is correct.

Record:

- eager versus compiled output;
- first-call compile time;
- warmed steady-state time;
- graph breaks and their reasons;
- shape/dtype/device specializations;
- recompilation count;
- backward behavior;
- memory changes.

For library or custom-operator development, a strict full-graph diagnostic mode is useful because silent graph fragmentation can hide the reason a speedup did not appear.

### 5.6 Export is not compilation and compilation is not deployment

Keep these boundaries distinct:

- **eager execution** runs operations immediately;
- **training/autograd graph** records derivative relationships;
- **compiled graph** transforms a region for a backend;
- **exported graph** creates a constrained, serializable program representation;
- **runtime graph** is what a deployment engine finally loads and partitions;
- **device kernel** is the code that runs on the selected hardware.

A result that works in one boundary does not prove the others. Test each transition with the same named outputs.

### 5.7 Portability is layered

There is no single portability switch.

- ISO C/C++ gives source portability but not equal vectorization or library performance.
- BLAS-like interfaces give algorithmic portability but allow different layout and numerical behavior.
- framework custom operators give orchestration portability only when every target backend is registered.
- ONNX gives graph interchange only for representable, versioned semantics.
- HIP provides a CUDA-aligned C++ path for AMD and NVIDIA ecosystems, but hardware-specific tuning still matters.
- Triton can shorten development for a class of GPU kernels, but it is not a replacement for every runtime, device, or low-level feature.

State the layer at which portability is claimed.

### 5.8 The first useful capability milestone

A reader has crossed the bridge when they can:

1. write a small, contract-explicit C/C++ numerical component;
2. generate an independent NumPy or framework reference;
3. verify forward, backward, and state updates as applicable;
4. expose the component through a correct framework boundary;
5. profile single-thread CPU, controlled multicore CPU, and GPU timing scopes;
6. decide from evidence whether to keep C, call a library, use a compiler, or write a backend-specific kernel;
7. reproduce the result from a clean environment and make a bounded claim.

That capability is more useful than superficial familiarity with a large number of AI tools.


## 6. Choose a practice environment before buying hardware

A practical bridge requires somewhere to run experiments. Do not treat access to a large GPU as the admission ticket. Most of the important transition work—tensor contracts, reference fixtures, autograd, native bindings, build systems, sanitizer runs, thread-pool behavior, and profiler discipline—can be learned on a CPU.

Choose an environment for the question being tested, not for prestige or peak FLOPS.

### 6.1 What to learn on CPU first

A local or cloud CPU environment is sufficient for:

- C/C++ build, warnings, sanitizers, and debugger use;
- NumPy and PyTorch tensor metadata, views, broadcasting, and reductions;
- eager autograd and finite-difference checks on tiny fixtures;
- C ABI, pybind11, or custom-operator boundary tests;
- optimizer/state traces and checkpoint round trips;
- compiler vectorization reports, cache behavior, and multicore scaling;
- CI, artifact capture, and reproducibility manifests;
- ONNX export and CPU-runtime comparison for small models.

This is not a consolation path. It is the shortest path to separating numerical, integration, and GPU-specific failures.

### 6.2 An environment ladder

| Environment | Best use | Main limitation | Evidence to preserve |
|---|---|---|---|
| Existing local CPU machine | Reference kernels, compilers, sanitizers, NumPy/PyTorch CPU, build/test work | May not expose GPU execution or datacenter CPU behavior | compiler, flags, CPU model, thread settings, package lock |
| Reproducible cloud CPU workspace | A clean Linux build, dev-container practice, CI-like reproduction | Session and storage quotas; usually not a GPU laboratory | image/dev-container revision, machine size, package lock |
| Hosted notebook with optional accelerator | First tensor/autograd experiments and short GPU comparisons | Ephemeral state, dynamic quotas, variable accelerator model, notebook-order hazards | exported script/notebook, runtime fingerprint, accelerator model |
| Local Windows plus WSL | Linux-native tools while retaining a Windows workstation; local GPU when supported | Driver/runtime boundary and vendor-specific support | Windows build, WSL version, driver, runtime, framework build |
| Free demo hosting | Small interactive inference or visualization demos | Not a substitute for a general native build/debug machine or long training run | repository revision, hardware tier, startup/runtime limits |
| Paid burst GPU | Controlled profiling, larger memory experiments, specific GPU architecture | Billing, idle storage, data movement, availability, and shutdown risk | exact SKU, image, driver, billing interval, start/stop times |
| CI runner | Repeatable CPU builds, sanitizers, numerical fixtures, packaging | Time and resource limits; performance numbers are often noisy | workflow revision, runner image, logs, artifacts |

### 6.3 Useful current options and their proper role

The service names below are examples, not endorsements. Availability, quotas, hardware, and prices change; always verify the current provider page before planning an experiment.

#### Local CPU and a project-local environment

Start here when possible. Install a C/C++ compiler, CMake, Python, NumPy, and CPU PyTorch in a project-local environment. Keep one command that prints the interpreter path, package versions, compiler version, CPU model, and active thread-related environment variables.

A small local CPU fixture is usually a better correctness oracle than a remote GPU notebook because it is easier to inspect, rerun, and freeze.

#### Browser notebooks

Google Colab and Kaggle Notebooks are useful for quick experiments without local setup. Colab explicitly uses dynamic limits: accelerator type, availability, idle timeout, and maximum runtime can change, and resources are not guaranteed. Kaggle exposes optional accelerators through notebook settings. Treat both as temporary execution environments rather than the only copy of the experiment.

Colab can also connect its notebook interface to a local runtime. This is a useful bridge when the reader wants notebook inspection while keeping data, compilers, or a local GPU under local control.

Required discipline:

```text
save code and fixtures outside ephemeral runtime storage
record the assigned accelerator before interpreting timing
restart and run all cells before treating the notebook as evidence
repeat the decisive test from a clean script when practical
```

#### Reproducible cloud CPU development

GitHub Codespaces provides a cloud development environment defined by repository configuration and a dev container. Personal accounts include a monthly usage allowance, but compute and storage remain metered resources. It is useful for a clean Linux compiler/toolchain exercise and for proving that the project can be reconstructed from files rather than from one workstation.

Use Codespaces as a development and reproduction environment, not as a default answer to every performance question.

#### Free hosting and shared GPU demos

Hugging Face Spaces provides a free default CPU environment for small applications, and ZeroGPU allows users to run supported shared-GPU Spaces subject to quotas and queueing. Spaces are well suited to demonstrations, visual inspection, and small inference tools. They are not the natural first environment for sanitizer-heavy native development or unrestricted long-running training.

#### Local GPU on Windows

WSL provides a practical route from a Windows workstation to Linux-oriented ML tools. Microsoft documents NVIDIA CUDA in WSL and PyTorch-DirectML for supported DirectX 12 hardware. Treat each framework/backend combination as version-sensitive. This route is especially useful to experienced Win32 developers, but the driver, Windows build, WSL layer, framework build, and GPU backend must all be recorded as part of the executable system.

#### Paid burst compute

Short-lived rented compute can be cheaper than purchasing hardware when the question is narrow: for example, whether a kernel crosses over at a certain shape, whether a model fits in a given memory size, or how a specific GPU architecture behaves.

Managed notebook/workspace services and on-demand GPU platforms differ in billing granularity, persistence, queueing, root/container access, networking, and idle charges. Current examples include Lightning AI Studios, Modal, Hugging Face Jobs, and Runpod. Do not copy a headline hourly price into a long-lived guide; compare the live total-cost contract instead.

### 6.4 The practice sequence

A compact progression should produce evidence at every step.

1. **Local scalar baseline** — compile one small C kernel, run strict warnings and sanitizers, and save deterministic output.
2. **Array-programming comparison** — express the same operation in NumPy; inspect shape, dtype, strides, and broadcasting; compare against the C fixture.
3. **Autograd comparison** — express it in PyTorch CPU, verify forward output and gradients, and observe gradient accumulation explicitly.
4. **Native boundary** — call the verified C/C++ path from Python or register it as an operator; test non-contiguous input, ownership, mutation, and error handling.
5. **Clean remote reproduction** — rebuild and run the fixture in a dev container, Codespace, or CI runner without relying on workstation state.
6. **Hosted GPU experiment** — move only the already-verified operation to a GPU notebook; separate transfer, warm-up, launch, synchronization, and steady-state time.
7. **Paid architecture-specific experiment, only if needed** — rent the exact GPU class required by the unanswered question, collect profiler artifacts, then terminate the resource.

Treat each step as a separate session with an exit criterion: a saved command, a captured environment, and a fixture result that still passes before the next layer is added.

### 6.5 Cost and cleanup are part of the experiment contract

Before starting any metered environment, write down:

```text
provider and region
exact CPU/GPU type, memory, and machine shape
billing unit and minimum charge
whether stopped/sleeping resources still incur storage cost
persistent versus ephemeral paths
network ingress/egress policy
preemption or maximum-session policy
container/root/compiler access
secret and dataset policy
automatic shutdown or spending cap
artifact copy-out command
resource deletion check
```

A cheap GPU with expensive idle storage, data transfer, or forgotten runtime can be the more expensive experiment. A free session that changes hardware between runs may be unsuitable for a performance claim even when it is excellent for learning an API.

### 6.6 When to buy hardware

Purchase hardware after repeated experiments show a stable need, not before the first tutorial.

A local GPU becomes easier to justify when several of these are true:

- the workload is run frequently enough that rental setup and transfer dominate;
- data cannot be moved to a third-party service;
- low-latency iteration matters every day;
- the required GPU class is available locally at acceptable power/noise cost;
- the software path is already validated on rented or shared hardware;
- the expected lifetime workload has been measured rather than guessed.

Until then, local CPU plus short accelerator sessions usually teaches more per unit of cost.

### 6.7 Use one fixture across every environment

A practice environment is useful only if the work can move without changing the numerical question. Use one deliberately small dense-layer plus stable-cross-entropy fixture across C, NumPy, optional PyTorch, local tests, CI, and optional GPU timing.

A minimal public exercise needs only ordinary files:

```text
fixture.json or fixed constants
reference.c
reference_numpy.py
optional_reference_torch.py
test command
captured environment summary
```

The correctness path is:

```text
explicit storage and reduction contract
→ scalar double forward/backward
→ centered finite differences
→ machine-readable native output
→ independent NumPy comparison
→ optional PyTorch autograd comparison
→ strict and sanitizer builds
```

The GPU exercise is separate. Record the assigned device and distinguish:

- host-to-device transfer;
- synchronized end-to-end time;
- device-event kernel time;
- steady-state pipelined time.

This prevents a hosted accelerator session from redefining the verified CPU contract. The verification work behind this guide may use larger test assets, but the public method does not require downloading a companion repository.

---

# Part II — Contracts and verification

## 7. The six contracts

A reliable AI numerical component should state six contracts before optimization.

### 7.1 Mathematical contract

Record:

- the exact equation;
- reduction axes;
- sum versus mean;
- normalization denominator;
- log base;
- epsilon placement;
- clipping and regularization order;
- overwrite versus accumulate behavior.

Example:

```text
loss              = mean negative log likelihood over valid tokens
logarithm          = natural logarithm
ignored positions  = excluded from numerator and denominator
output gradient    = softmax(logits) - one_hot(target)
parameter gradient = accumulated over microbatches, divided once at update
```

“Cross-entropy,” “RMSprop,” or “layer normalization” is not a complete specification. Small variant differences can materially change output.

### 7.2 Tensor metadata contract

For every input and output, record:

```text
semantic axes
shape
strides
contiguous dimension
layout or packing
storage dtype
compute dtype
accumulation dtype
device
alignment
legal broadcasting
```

Use named axes in documentation even when the implementation stores flat memory.

```text
activation[batch][time][channel]
weight[out_channel][in_channel]
```

For matrix multiplication, state whether the conceptual relation is:

```text
C[M,N] = A[M,K] × B[K,N]
```

and separately describe how each matrix is physically stored and passed to the backend.

### 7.3 Mutation and aliasing contract

Record:

- which inputs may be modified;
- whether outputs are fresh allocations, views, or aliases;
- legal in-place operation;
- overlapping-buffer rules;
- ownership and lifetime.

This contract is central to modern framework integration. PyTorch custom operators, for example, require mutation and aliasing behavior to be declared rather than inferred from an arbitrary kernel body.

### 7.4 State and lifetime contract

Keep these categories distinct:

| State class | Examples | Typical lifetime |
|---|---|---|
| Parameters | weights, biases, learned scales | model lifetime |
| Gradients | `dW`, `db`, accumulated partials | accumulation/update interval |
| Optimizer state | moments, step count | training run and resume |
| Running statistics | batch-normalization mean/variance | mode and checkpoint dependent |
| Sequence/cache state | recurrent state, key-value (KV) cache, delay line | stream or request policy |
| Workspace | im2col buffers, saved activations | operation or forward/backward pair |
| RNG state | sampling, dropout, initialization | experiment and checkpoint policy |
| Diagnostic state | counters, moving averages | reporting interval |

Resetting one class at the boundary of another is a common source of silent error.

### 7.5 Execution and backend contract

Record:

- CPU, GPU, or accelerator;
- thread and stream behavior;
- synchronization points;
- deterministic versus opportunistic algorithms;
- legal concurrency and reentrancy;
- workspace ownership;
- backend version and algorithm choice;
- fallback behavior.

### 7.6 Experiment contract

A seed alone is not a reproducibility specification. Record:

```text
source revision
compiler and version
build preset and flags
framework/runtime versions
CPU/GPU and instruction capability
math, BLAS, and backend libraries
thread and stream settings
dataset and preprocessing identity
RNG algorithms and seeds
model shapes and dtypes
initialization and optimizer variant
checkpoint schema
comparison tolerance
```

Store this manifest beside the fixture output or benchmark result. It should be ordinary text that can be reviewed without the original interactive session.

## 8. Evidence levels

Use evidence labels to avoid stronger conclusions than the tests support.

| Level | Evidence | Supported claim |
|---|---|---|
| E0 | code reading only | a plausible hypothesis |
| E1 | clean build and reviewed warnings | limited static confidence |
| E2 | sanitizer or runtime-check pass | no covered memory/UB failure observed |
| E3 | deterministic forward fixture | local forward agreement for tested shapes |
| E4 | finite difference plus independent backward comparison | local gradient confidence |
| E5 | fixed optimizer/state trace and serialization round trip | state-transition confidence |
| E6 | scalar versus optimized differential tests | backend-equivalence confidence for covered cases |
| E7 | repeated end-to-end runs and deployment tests | integration and robustness evidence |

Examples:

- A decreasing loss is integration evidence, not proof that every gradient is correct.
- A sanitizer-clean run does not detect a safe traversal of the wrong axis.
- Agreement with a framework is stronger when finite differences or a second implementation also agree.
- A fast benchmark says nothing about numerical equivalence unless the same case passed the correctness gate.

## 9. Build a tiny executable reference before a large model

The most useful fixture is not “realistic.” It is discriminating.

### 9.1 Use deliberately unequal dimensions

Square and symmetric shapes hide transposes and swapped bounds.

Prefer shapes such as:

```text
M = 2
K = 3
N = 4
batch = 2
sequence = 3
channels = 5
```

### 9.2 Use diagnostic values

Good fixture values expose position and sign:

```text
0.10, -0.20, 0.35, -0.45, 0.60
```

or structured integers that remain exactly representable over the tested range.

Avoid initially:

- all zeros;
- all ones;
- symmetric matrices;
- identical channels;
- values near non-differentiable boundaries;
- large random tensors that are hard to inspect.

### 9.3 Compare the first divergence

For an operator chain, capture named stages:

```text
input validation
layout conversion
preactivation or accumulation
activation
normalization statistic
loss contribution
incoming gradient
parameter gradient
updated state
```

The first divergence usually identifies the broken contract. Final loss often does not.

### 9.4 Use a double-precision scalar oracle where practical

A small `double` implementation is useful because it is:

- independent of many optimized paths;
- easy to print and inspect;
- stable enough for finite differences;
- portable across C, C++, Python, and notebooks.

It is still not absolute truth. Its equations, axes, and reductions must also be tested.

### 9.5 Make the reference executable and machine-readable

A useful reference is not only a code listing. It should have:

```text
one configure/build/test command
fixed fixture values
human-readable pass/fail output
machine-readable intermediate tensors
independent comparison script
recorded compiler/interpreter/backend identity
```

A compact implementation can use CMake Presets, CTest labels, JSON or another simple tensor dump, a NumPy comparison script, and an environment fingerprint. The same fixture can then be inspected manually, run in CI, or consumed by a notebook without making the notebook the authority.

This structure also makes assisted review safer: a human or coding agent can change one branch, rerun the same commands, and inspect the first divergence rather than relying on a prose claim that the code was tested.

## 10. Tensor semantics: shape is only the start

### 10.1 Shape and semantic axes

Do not document only integer dimensions. Name the meaning of each axis.

```text
x[B,T,C]
y[B,T,O]
W[O,C]
```

A shape-compatible transpose can remain numerically plausible. Semantic names expose that class of bug.

### 10.2 Strides and non-contiguous views

A tensor view may have the expected shape but unexpected strides or storage offset.

Before passing a tensor to native code, decide whether the kernel:

- accepts arbitrary strides;
- accepts a limited set of layouts;
- requires a contiguous copy;
- writes into a view;
- assumes alignment beyond what the caller guarantees.

A binding that passes `data_ptr()` while silently ignoring strides is correct only for the accepted contiguous layout.

### 10.3 Broadcasting

Broadcasting is not a C pointer feature; it is an operator rule.

Tests should include:

- equal shapes;
- scalar broadcast;
- singleton-axis broadcast;
- incompatible shapes;
- backward reduction over broadcast axes.

A forward broadcast can look correct while the gradient incorrectly omits the required reduction.

### 10.4 Dtype and promotion

Record:

- input storage dtype;
- internal compute dtype;
- accumulator dtype;
- output dtype;
- scalar-literal conversion;
- integer overflow and saturation policy.

Never infer accumulator width from output width. INT8 matrix multiplication commonly requires a wider accumulator; FP16 or BF16 storage commonly uses FP32 accumulation in selected operations.

### 10.5 Device and ownership

A valid pointer on one device is meaningless on another. Record whether the memory is:

- host pageable;
- host pinned;
- device;
- unified/managed;
- shared with a framework allocator;
- valid only until the next operation or stream event.

### 10.6 Layout conversion is an operator

Transpose, reorder, pack, and dequantize steps consume time, memory, and numerical assumptions. Treat them as named operators with tests and benchmark scope, not invisible glue.

### 10.7 Treat preprocessing as a versioned operator graph

The model does not begin at its first learned layer. Decoding, resampling,
normalization, tokenization, windowing, augmentation, and label construction are
numerical operators. A deployment that uses different preprocessing can be
wrong while every model tensor and weight is correct.

Give each sample enough identity to reproduce its path:

```text
sample ID and source revision
preprocessing graph/version
units and numeric range
channel or feature order
resize/crop/window policy
normalization constants and dtype
random seed or augmentation parameters
label/vocabulary/schema version
```

Do not encode this contract only in filenames or notebook state. Save it with
the experiment and deployment artifact. For a reference fixture, choose samples
whose results change if a plausible alternative is used: RGB versus BGR,
`[0,255]` versus `[0,1]`, center crop versus stretch, inclusive versus exclusive
window end, or a different tokenizer normalization rule.

A useful data differential test compares:

```text
raw sample
→ independent reference preprocessing
→ framework preprocessing
→ native/deployment preprocessing
```

Compare the first model-ready tensor, not only the final prediction.

### 10.8 Collation creates a new ownership boundary

A sample object and a batch object have different lifetimes. Make collation an
explicit operator that produces a new owner for:

- batched values;
- lengths or offsets;
- masks;
- labels and sample IDs;
- optional sort and inverse-sort mappings;
- preprocessing identity.

Do not let a batch retain untracked views into decoder scratch space, a reused
ring slot, a memory-mapped window that may close, or a worker-local temporary.
If zero-copy is intentional, record the original owner and the event or scope
that ends the borrow.

Test collation with unequal lengths and dimensions. Mutate or release the source
samples after collation and verify that an owning batch is unchanged. If the
batch is intentionally a view, make the opposite test: prove that aliasing is
visible and that the owner outlives every consumer.

### 10.9 Variable length needs three independent contracts

Padding, lengths, and masks are related but not interchangeable.

1. **Lengths or offsets** identify the valid extent of each sample.
2. **Padding values** initialize unused storage. They are not generally semantic.
3. **Masks** encode which positions an operator may read or contribute.

Write mask polarity, shape, dtype, and broadcast axes. For example:

```text
values[B,T,C]
lengths[B]
valid[B,T] where true means data is valid
loss_mask[B,T] where true means the item contributes to the denominator
attention_mask[B,1,1,T] where the operator-specific convention is documented
```

Framework APIs do not all use the same polarity. Some boolean attention masks
use `true` to mean “blocked,” while application masks often use `true` to mean
“valid.” Convert at a named boundary and test the conversion.

For a masked mean, the denominator is the number or total weight of valid
contributions, not `B*T`:

```text
sum(value * valid) / sum(valid)
```

The fixture should remain unchanged when the padding sentinel changes. A naive
unmasked reduction should fail the fixture. Also test:

- unsorted lengths and restoration to original sample order;
- a length of one and the maximum supported length;
- non-square `B`, `T`, and `C`;
- all-masked or zero-length input according to the declared reject/empty policy;
- truncation, bucket boundaries, and a length beyond allocated storage;
- loss normalization across uneven batches and distributed workers.

Packed representations are new layouts, not merely compressed arrays. Preserve
and test their batch-size metadata, sort mapping, inverse mapping, and unpacked
padding policy.

### 10.10 Worker processes own resources, not borrowed parent state

A data-loader worker is a separate ownership domain. Decide where these are
created and destroyed:

- file descriptors, archive readers, database/network clients, and decoder contexts;
- random-number generators and augmentation state;
- caches and memory maps;
- thread pools used inside decoding libraries;
- accelerator runtime state.

Do not assume that copying a Python object or forking a process gives each worker
an independent, safe native resource. Initialize worker-local resources in the
worker, shard iterable sources explicitly, and derive worker seeds from a
recorded base seed plus stable worker/epoch identity.

With accelerators, process start method is part of the runtime contract. Current
PyTorch multiprocessing guidance warns about initializing an accelerator before
a `fork`-based child (“poison fork”) and recommends a compatible start method
such as `spawn` or `forkserver` where required. This is a correctness issue
before it is a throughput tuning issue.

Treat queue depth as bounded memory, not a magic speed knob:

```text
resident batches ≈ workers × prefetched batches per worker × batch bytes
```

The exact framework implementation may add more buffers. Measure process RSS,
shared-memory use, pinned-memory use, queue occupancy, consumer idle time, and
worker restart behavior. Test end-of-epoch, early cancellation, worker failure,
and a slow consumer so leaked or overwritten buffers become visible.

### 10.11 Pinned memory and non-blocking copies require completion proof

Pinned host memory is a limited transfer resource. It can improve host-to-device
transfer behavior, but pinning pageable memory is itself work and can reduce
system flexibility. Pin at a deliberate stage—often after collation—and measure
whole-pipeline throughput rather than assuming that more pinned buffers are
better.

`non_blocking=True` or an asynchronous copy call means that the host may return
before transfer completion. It does not mean that the destination is ready or
that the source can be modified or recycled. Record:

```text
source owner and storage class
destination device and stream
copy-enqueue point
completion event or synchronization rule
first legal destination read
first legal source reuse
```

Current PyTorch guidance shows that asynchronous CPU-to-device copies can be
useful, but manually calling `pin_memory()` immediately before transfer can be
slower, and device-to-CPU non-blocking copies require synchronization before a
CPU consumer reads the result. Treat these as benchmarked, direction-specific
contracts rather than universal switches.

### 10.12 Side streams extend allocator lifetime

A framework allocator often knows the stream on which storage was allocated, not
every stream that later uses it. When a tensor is consumed on another stream,
make lifetime visible through the framework's stream-recording facility or an
explicit event/lease protocol.

In PyTorch, `Tensor.record_stream(stream)` tells the caching allocator that the
tensor has work pending on that stream. It is a lifetime aid, not a general data
dependency primitive: operations still need the correct stream waits, and an
explicit event-based owner can sometimes be more precise. Test early deletion,
rapid allocator reuse, multiple side streams, exceptions, and cancellation.

### 10.13 Zero-copy interchange is an ownership protocol

“Zero-copy” does not mean “no lifetime problem.” It usually means that two
objects can address the same storage.

For every interchange boundary, record:

- producer and consumer;
- shape, strides, dtype, device type, and device identifier;
- whether a copy is forbidden, allowed, or required;
- who owns the storage and who invokes the deleter;
- mutation/aliasing policy;
- stream handoff and completion rule.

DLPack makes these responsibilities explicit. Its Python protocol requests a
zero-copy view when possible, passes the consumer stream to the producer for
necessary synchronization, and transfers a managed tensor to one consumer. A
raw capsule is consumed exactly once; the consumer assumes responsibility for
the managed object's deleter. Test one-time consumption, mutation visibility,
strides, device identity, and owner destruction. Do not infer safety merely from
successful conversion.

### 10.14 Data-pipeline acceptance fixture

A compact acceptance fixture should prove, on CPU before accelerator tuning:

```text
one preprocessing identity
unequal-length deterministic samples
new owning contiguous batch or explicit borrow
length/mask/padding agreement
valid-element denominator
padding-sentinel invariance
wrong-polarity and invalid-length rejection
sort/unsort round trip for packed layouts
bounded buffer lease and backpressure
zero-copy alias and ownership behavior where used
```

Then add environment-specific evidence for workers, real decoders, pinned memory,
streams, transfer events, and failure recovery. Keep each unexecuted branch
marked `NOT_RUN`; a CPU mask fixture does not prove an asynchronous GPU pipeline.

## 11. Verify operator families, not only whole models

A small set of operator families covers most numerical failure modes.

### 11.1 Elementwise transforms

Examples:

- activation functions;
- affine scale and bias;
- clipping;
- masking;
- quantize/dequantize.

Check:

- special values and boundary values;
- NaN/Inf policy;
- in-place aliasing;
- derivative at representative smooth points;
- vector tail handling.

### 11.2 Reductions

Examples:

- sum and mean;
- norms;
- softmax denominator;
- normalization statistics;
- dot products.

Check:

- axis and denominator;
- empty or masked inputs;
- accumulator precision;
- overflow/underflow;
- parallel reduction order;
- deterministic versus non-deterministic behavior.

### 11.3 Matrix multiplication and convolution

Check:

- conceptual `M`, `N`, `K` mapping;
- destination/source orientation;
- leading dimensions and strides;
- transpose flags;
- bias placement;
- batch dimensions;
- padding, dilation, groups, and channel layout for convolution;
- temporary packing and reorder costs.

A canonical hand-written matrix relation is:

```c
for (size_t m = 0; m < M; ++m)
    for (size_t n = 0; n < N; ++n)
        for (size_t k = 0; k < K; ++k)
            C[m * ldc + n] += A[m * lda + k] * B[k * ldb + n];
```

This is a reference relation, not a performance recommendation.

### 11.4 Stable softmax and cross-entropy

Do not compute `log(softmax(x))` through an avoidable unstable intermediate.

For logits `x_i`, use a shifted log-sum-exp:

```text
m       = max_i x_i
logsum  = m + log(sum_i exp(x_i - m))
loss    = logsum - x_target
```

Test:

- a uniform vector;
- one very large logit;
- one very negative logit;
- batch and mask denominators;
- target bounds;
- gradient sum near zero across classes.

The runnable example `examples/stable_cross_entropy.c` demonstrates the scalar principle.

### 11.5 Normalization

Normalization combines reduction, epsilon policy, saved state, and mode behavior.

Record:

- normalized axes;
- biased or unbiased variance;
- epsilon placement;
- training versus inference statistics;
- accumulation dtype;
- running-statistics update rule;
- backward reduction axes.

Do not treat “batch norm,” “layer norm,” or “RMS norm” as interchangeable labels.

## 12. Backward computation and automatic differentiation

### 12.1 A backward pass is a routing and accumulation program

For every input and parameter, write the incoming-gradient equation before coding the loop.

If a value contributes to multiple downstream operations, its gradient is normally the sum of those routes.

Common failures include:

- overwrite instead of accumulation;
- missing broadcast reduction;
- wrong transpose in input gradient;
- correct local derivative applied to the wrong saved value;
- missing initial-state or parameter route;
- reduction performed twice or not at all.

### 12.2 Use three-way triangulation

When practical, compare:

1. the hand-written analytic gradient;
2. centered finite differences in double precision;
3. an independent autodiff implementation.

No one path is infallible. Agreement across independent mechanisms is much stronger.

### 12.3 Centered finite differences

For scalar objective `L` and parameter `θ_i`:

```text
g_num = [L(θ_i + ε) - L(θ_i - ε)] / (2ε)
```

Use an epsilon sweep rather than one magic constant. A useful sweep for double-precision fixtures is often:

```text
1e-2, 1e-3, 1e-4, 1e-5, 1e-6
```

Interpret the curve:

- error falls, then rises: expected truncation/rounding trade-off;
- no stable region: likely contract, scale, or non-smoothness problem;
- only a few indices fail: likely indexing or missing route;
- all values differ by a constant factor: likely reduction denominator.

Avoid testing exactly at ReLU kinks, clipping thresholds, ties in `max`, or quantization discontinuities.

### 12.4 Gradient comparison metric

A common scale-aware measure is:

```text
|g_analytic - g_numeric| / max(1, |g_analytic|, |g_numeric|)
```

For very small values, also inspect absolute error. Tolerance should be tied to dtype, conditioning, operation depth, and backend.

### 12.5 Framework gradcheck and custom-op validation

Framework tools can validate more than raw numbers. Current PyTorch custom-operator guidance emphasizes explicit schemas, representative examples, fake/meta behavior, and `torch.library.opcheck` for composition with framework subsystems.

Treat these checks as integration evidence. They do not replace a transparent scalar fixture for the kernel itself.

## 13. Verify state transitions independently

Training and stateful inference add programs around the kernel.

### 13.1 Optimizers

An optimizer name is not a full contract. Record:

- gradient sum or mean;
- clipping order;
- weight decay form and order;
- momentum or moment equations;
- epsilon placement;
- bias correction;
- step counter update;
- parameter traversal order;
- state initialization and serialization.

Use a fixed external gradient trace. After each step, compare:

```text
parameter
first moment or momentum
second moment
step counter
```

The runnable `examples/adam_trace_reference.c` is deliberately independent of a model.

### 13.2 Running statistics and caches

Test stateful components with:

- one call;
- repeated identical calls;
- reset then repeat;
- save/load then next call;
- interleaved independent streams;
- training versus inference mode.

### 13.3 Sequence state

Recurrent state, streaming convolution history, and KV caches share a core rule:

> State carry, gradient carry, cache validity, and update timing are separate decisions.

A recurrent cell is useful here because it forces the implementation to state exactly when state is carried, detached, reset, serialized, and differentiated.

### 13.4 Treat attention and KV caching as one contract stack

Attention is not one matrix multiplication. A compact causal block combines
layout, batched multiplication, scaling, masking, stable reduction,
normalization, residual addition, precision, and state ownership:

```text
Q,K,V: [batch, heads, tokens, head_dimension]
scores = Q @ transpose(K) / sqrt(head_dimension)
probabilities = stable_masked_softmax(scores, allow)
attention = probabilities @ V
output = input + output_projection(attention(layer_norm(input)))
```

Write the meaning of every axis. Do not rely on a framework's preferred layout
or on a square test where query and key lengths happen to match. Use unequal
batch, head, query, key, and feature dimensions in the transparent fixture.

#### Mask polarity and position are separate contracts

A useful explicit boundary is:

```text
allow[B,H,L,S] where true means key position S may participate
```

Convert API-specific masks at one named boundary. Current PyTorch 2.13
[`scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
documentation uses `True` to mean *participates* for its boolean `attn_mask`,
but documents the opposite polarity for `MultiheadAttention.key_padding_mask`.
That is a conversion obligation, not a detail to remember informally.

Causality also needs absolute positions. With cached decoding, a query block may
have length `L=2` while the combined K/V cache has length `S=5`. If the queries
represent absolute positions `[3,4]`, the allowed rows are:

```text
q=3: [1 1 1 1 0]
q=4: [1 1 1 1 1]
```

A generic upper-left `2 x 5` triangle instead permits only `[0]` and `[0,1]`.
PyTorch currently documents upper-left alignment for non-square
`is_causal=True`; do not assume that flag encodes the offset required by a
particular cache. Pass or construct an explicit position-aware mask when the
API contract requires it.

Define the all-masked-row policy. Rejecting the call, returning zeros, or using
a sentinel are different contracts. A stable reference should subtract the
maximum allowed score, set blocked probabilities exactly to zero, and verify
that every accepted row is finite and sums to one.

#### Cache update timing is part of the equation

For ordinary autoregressive self-attention, the current token's K/V normally
participate in the current query. A safe step is:

```text
normalize and project current input
→ append current K/V to this request's cache
→ construct mask from absolute query/key positions
→ evaluate attention over past plus current K/V
→ project and add the residual
→ publish the new cache length
```

Appending after attention computes a different model. So do stale cache reuse,
resetting at an internal chunk boundary, or sharing one mutable cache between
independent requests.

Treat each cache as an owned state object with:

```text
request or stream identity
layer identity
batch/head/feature layout
valid length and capacity
position convention
storage dtype/device
append, reset, save/resume, and overflow policy
completion event when updates are asynchronous
```

Test a dynamic append cache and a fixed-capacity cache against the same full
causal reference. Dynamic storage is simple but changes shape and may reallocate.
Static storage keeps shape stable but consumes declared capacity and may spend
work on masked slots. Current Hugging Face
[cache documentation](https://huggingface.co/docs/transformers/kv_cache)
exposes the same broad trade-off; model-specific cache classes and policies
remain version-sensitive.

#### Preserve the pre-norm, residual, and precision contracts

Layer normalization usually reduces over the feature axis for each token, not
over time or batch. The residual adds the original block input to the projected
attention result; omitting it can leave tensor shapes valid while changing the
model completely.

Record storage, compute, accumulation, and output dtypes separately. Current
PyTorch SDPA documentation notes that fused backends may differ numerically and
that its math path keeps intermediates in float for half/BF16 inputs. Use a
float64 or double reference, compare the selected backend over a shape matrix,
and replay at least one decision-level metric. Do not infer accelerator behavior
from a CPU math-path result.

Functional SDPA also applies a nonzero `dropout_p` even when the surrounding
module is in evaluation mode. Pass `0.0` explicitly for inference. Training
with attention dropout, backward checks, and cache interaction is a separate
branch; cache-based generation should not be presented as a training shortcut.

#### Minimum attention/cache acceptance

A compact integration fixture should discriminate at least:

- scalar or transparent attention versus the candidate operator;
- mask polarity, padding, causality, and an all-masked row;
- square full execution versus non-square cached decoding;
- several chunk partitions versus one full causal call;
- current-K/V omission, stale reset, capacity overflow, and wrong-axis mutations;
- two interleaved streams with separate caches;
- normalization axis and residual addition;
- higher-precision accumulation versus a reduced-precision mutation;
- tensor tolerance plus task/decision replay;
- backend and accelerator branches marked `NOT_RUN` until measured.

A bounded CPU reference experiment for this guide passed **26/26** checks, including an
independent scalar oracle, PyTorch CPU SDPA differential comparison, dynamic and
static cache equivalence, non-square causal-mask discrimination, update timing,
stream isolation, normalization/residual mutations, and float16-storage versus
float32-accumulation behavior. CUDA fused kernels, long-context production
caches, GQA/MQA, backward, export, and throughput remain `NOT_RUN`.

### 13.5 RNG state

Randomness affects:

- initialization;
- dropout and stochastic layers;
- sampling;
- shuffling;
- augmentation;
- parallel worker behavior.

Record the algorithm and state, not only a seed integer. Exact resume may require serializing multiple RNG states and the data cursor.

## 14. Treat one training step as a state machine

A training step is not merely `forward`, `backward`, and `update`. It is a
state transition whose ordering determines the mathematical result.

A useful effective-batch model is:

```text
acquire the next identified samples
→ enter training mode
→ clear gradients at the accumulation-window boundary
→ run forward and form a loss numerator and denominator
→ accumulate backward contributions for every microbatch
→ unscale gradients, if loss scaling is active
→ reject non-finite gradients or mark the update as skipped
→ inspect or clip the unscaled gradients
→ apply the optimizer update once
→ update the loss scale and learning-rate schedule at their defined boundaries
→ clear or retain gradients according to the next accumulation window
→ advance the data cursor and persistent step state
```

Changing this order can create a different algorithm while every individual
operator still appears correct.

### 14.1 Define the commit boundary

Treat `optimizer.step()` as a commit operation. Before it, parameters are old
and gradients describe one declared effective batch. After it, parameters,
optimizer moments, step counters, scheduler state, and possibly the loss scale
belong to the next state.

Write the boundary explicitly:

```text
optimizer_step
microbatch_index within the effective batch
effective sample or valid-element denominator
gradient scaling state
clipping threshold and norm definition
non-finite policy
scheduler update boundary
data cursor consumed by this update
```

Do not increment a public "step" counter for every forward call when the
optimizer updates only after several microbatches. Log both counters when both
matter.

### 14.2 Preserve the loss numerator and denominator across microbatches

Gradient accumulation is equivalent to a larger batch only when the accumulated
objective is equivalent.

For a loss reduced over valid elements, prefer the conceptual form:

```text
L_effective = sum(all microbatch loss numerators)
              / sum(all corresponding valid denominators)
```

Then each microbatch may backpropagate:

```text
microbatch_numerator / effective_denominator
```

This remains correct for unequal microbatch sizes and variable-length masks.
Averaging each microbatch mean equally is generally wrong when their
denominators differ. Dividing a loss that is already normalized by the number
of accumulation steps is also wrong unless that division is exactly how the
full effective-batch objective was defined.

A discriminating fixture should use unequal microbatches, such as two and three
samples, and compare both accumulated gradients and the resulting parameter
update against one full-batch execution.

### 14.3 Keep one gradient scale throughout an accumulation window

With automatic mixed precision, gradients produced from a scaled loss remain
scaled during accumulation. For one effective batch:

- keep the scale factor constant while microbatch gradients are added;
- do not unscale partway through the window;
- call `unscale_` at most once for each optimizer and only after accumulation is
  complete;
- call `step` and `update` at the effective-batch boundary, not for every
  microbatch.

Otherwise later scaled gradients are added to earlier unscaled gradients, or to
gradients produced with a different scale, and the original sum cannot be
recovered.

### 14.4 Unscale before inspection or clipping

Clipping a scaled gradient applies the threshold in the wrong units. The
required order is:

```text
backward on scaled loss
→ finish accumulation
→ unscale optimizer-owned gradients
→ test finiteness
→ measure or clip the unscaled gradients
→ optimizer step or explicit skip
→ scale update
```

Record both the pre-clip norm and whether an optimizer update actually occurred.
A loss scaler may skip the update when gradients contain `Inf` or `NaN`; the
training loop, scheduler, checkpoint metadata, and logs must define whether
they advance on a skipped update.

For more than one optimizer, each optimizer owns a separate unscale and step
decision. A single global boolean can hide partial updates.

### 14.5 Gradient clearing is part of optimizer semantics

Autograd accumulates into existing leaf gradients. Therefore, moving
`zero_grad` inside an accumulation window silently discards earlier
microbatches.

Also distinguish a missing gradient from a numerical zero. In current PyTorch
optimizer semantics, `set_to_none=True` leaves parameters that received no
gradient with `.grad is None`; many optimizers skip those parameters. An
explicit zero tensor can still trigger optimizer logic such as weight decay or
state updates.

Tests should cover:

- clearing exactly once at the declared window boundary;
- an intentionally unused parameter;
- `None` versus an explicit zero gradient;
- frozen parameters and conditional branches;
- stale gradients after an exception or skipped update.

### 14.6 Verify the optimizer with a fixed external gradient trace

A model-level loss curve is a weak optimizer test. Feed a small, fixed gradient
sequence into one parameter and compare every update against an independent
formula.

For Adam-like methods, retain after each step:

```text
parameter
first moment
second moment
bias-corrected values or equivalent update
step counter
learning rate and epsilon policy
```

This isolates optimizer equations from forward, backward, batching, and data
order. Then add an end-to-end test that proves the same optimizer state is
connected to the correct model parameters.

### 14.7 A resumable checkpoint must reproduce the next step

Saving weights is not the same as saving a training state. A practical
single-process checkpoint schema normally needs:

```text
schema and code/data contract versions
model parameters and registered buffers
optimizer state and parameter-group settings
scheduler state
mixed-precision scaler state
optimizer-step and microbatch counters
CPU and relevant accelerator RNG states
sampler or data cursor state
train/eval mode or an explicit restore rule
stateful model caches carried across batches
```

The strongest compact acceptance test is:

1. train to an optimizer boundary;
2. serialize the full declared state;
3. execute one more identified batch and retain the output, loss, update, and
   resulting state;
4. construct fresh objects, restore the checkpoint, and execute that same next
   batch;
5. require the declared replay target: bitwise equality where supported, or a
   stated numerical and task-level tolerance otherwise.

Deliberately omit optimizer state, RNG state, data cursor, and mode one at a
time. The fixture should fail for each omission. This proves that the recorded
fields are necessary rather than ceremonial.

Checkpoint at an optimizer boundary unless the schema also serializes all
partially accumulated gradients, the exact accumulation position, the current
loss scale, and the unconsumed data identity. Rejecting unsupported
mid-accumulation saves is safer than pretending they are resumable.

### 14.8 Persistence checkpointing and activation checkpointing are different

A persistence checkpoint stores state for restart. Activation checkpointing
saves memory during one forward/backward execution by discarding selected
intermediates and recomputing them during backward.

Recomputation adds contracts:

- the recomputed function must behave consistently with the original forward;
- random operations require an RNG-state policy;
- mutation, global state, I/O, or a changing cache can make recomputation
  incorrect;
- the selected checkpoint implementation may differ in graph recording and
  backward API support.

A useful stochastic fixture compares direct execution with checkpointed
execution under the same seed. With RNG preservation, forward values and
gradients should meet the declared agreement target. A mutation that disables
RNG preservation should change the backward gradient for a dropout-containing
region and be detected.

### 14.9 Training-step acceptance fixture

A compact CPU fixture can prove the protocol before accelerator-specific work.
A bounded CPU reference experiment for this guide covered:

- full-batch versus unequal-microbatch gradient and update agreement;
- equal-microbatch-mean, double-division, and inner-window `zero_grad`
  mutations;
- `None` versus zero-gradient optimizer behavior;
- scaled-gradient unscale, clipping order, and non-finite update skipping;
- an independent four-step Adam trace;
- a versioned checkpoint containing model, optimizer, scheduler, scaler, RNG,
  mode, counters, and data cursor;
- exact next-step replay after reconstruction;
- divergence when optimizer state, RNG, cursor, or mode is omitted;
- rejection of an unsupported mid-accumulation checkpoint;
- activation-checkpoint RNG recomputation behavior.

This does not prove mixed-precision convergence on an accelerator, distributed
synchronization, production sampler recovery, or crash-safe checkpoint
publication. Keep those claims separate until they are run in the target
environment.

## 15. Use several test styles

### 15.1 Example-based tests

A fixed input and expected output. Best for:

- exact contracts;
- regression fixtures;
- format conversion;
- optimizer traces;
- serialization.

### 15.2 Property tests

Examples:

- probabilities are finite and sum to one within tolerance;
- a norm is non-negative;
- zero gradient leaves an SGD parameter unchanged;
- quantized values stay in representable range;
- output shape follows the schema.

### 15.3 Differential tests

Compare:

- scalar C versus NumPy/PyTorch;
- float versus double reference;
- scalar versus BLAS/oneDNN;
- CPU versus GPU;
- unfused versus fused;
- full model versus exported runtime.

### 15.4 Metamorphic tests

Useful relations include:

- splitting a batch and recombining results;
- permuting independent batch items and undoing the permutation;
- adding a constant to all logits leaves softmax probabilities unchanged;
- full streaming input equals correctly carried chunked input;
- save/load preserves the next output or next update;
- scalar and vectorized paths agree within the declared tolerance.

Do not invent invalid properties such as “training loss must decrease every step.”

### 15.5 Mutation tests

Deliberately introduce realistic defects:

- swap two axes;
- omit one reduction;
- change `+=` to `=`;
- shift a packed channel block;
- reset state at the wrong boundary;
- increment optimizer step in the wrong place;
- ignore strides;
- use output dtype as accumulator dtype.

A test suite that survives these mutations is not discriminating enough.

## 16. Accept at operator, model-output, and task levels

A deployment candidate is not accepted merely because one operator is close to
a reference, and it is not accepted merely because one aggregate accuracy
number is unchanged. Use three distinct gates:

```text
operator and intermediate agreement
→ complete model-output agreement over a declared input envelope
→ task metric and decision-boundary acceptance
```

Each gate answers a different question. A compensating downstream error can
hide an incorrect intermediate. A small output difference can cross an
argmax, threshold, ranking, or greedy-decoding boundary. Conversely, unchanged
labels can hide severe drift in logits, probabilities, calibration, saturation,
or a minority slice. Required gates are conjunctive: passing one does not waive
another.

### 16.1 Define the three gates before optimization

| Gate | Primary question | Typical evidence |
|---|---|---|
| Operator/intermediate | Is the implemented numerical transformation the intended one? | diagnostic tensors, independent formulas, property and mutation tests |
| Model output | Does the complete candidate reproduce every declared output over the supported envelope? | shape/dtype/schema checks, absolute/relative error, per-output summaries, state replay |
| Task/decision | Does the candidate preserve the behavior that the application actually uses? | thresholded labels, top-k/ranking, sequence decisions, task metrics, calibrated confidence, slices |

Do not use a task score to debug the first divergent operator. Do not use a
local tensor tolerance as the only release gate for a system whose output is a
discrete or stateful decision.

### 16.2 Make the model-output envelope explicit

A complete output comparison needs more than one convenient example. Record:

```text
input and preprocessing version
sample or stream identifiers
dynamic shape and length matrix
mask, padding, and state initialization
input/output dtype and layout
all public output fields
absolute and relative tolerance policy
non-finite and saturation policy
state carry and reset boundaries
reference and candidate build fingerprints
```

Test the smallest, ordinary, boundary, and largest supported shapes. Summaries
should include more than a single maximum: retain the failing index, magnitude,
reference scale, and useful percentiles or counts. For multiple outputs, report
each output independently; a small auxiliary tensor must not disappear inside
a large aggregate.

Tolerance is a numerical contract, not a generic permission. It should be tied
to dtype, scale, reduction depth, backend, and the downstream decision. Exact
identity may be appropriate for integer metadata, shapes, masks, token IDs, or
a deterministic replay path even when floating outputs use tolerance.

### 16.3 Replay the real decision boundary

Near a boundary, a numerically small difference may have a large semantic
effect:

- two nearly tied logits can exchange `argmax`;
- `0.5001` and `0.4999` can produce opposite binary decisions;
- a small score change can alter top-k membership or ranking order;
- one changed greedy token can alter every later autoregressive input;
- a detection box can move across an IoU acceptance threshold;
- a control output can cross a clamp, dead band, or safety limit.

Therefore compare both continuous outputs and the exact downstream decision.
Include margin buckets: samples far from a boundary and samples near it should
not be treated as equally informative. If the threshold is tuned for a business
or safety metric, version the threshold, metric, label of interest, and
validation set separately from the trained weights.

### 16.4 Task metrics are necessary but not sufficient

The reverse failure is equally important. Two candidates can have identical
accuracy while one has much smaller margins, clipped probabilities, or large
logit drift. This can damage threshold portability, calibration, ranking,
uncertainty handling, later fine-tuning, or composition with another stage.

For classification, consider at least the outputs actually consumed:

- labels or top-k decisions;
- scores, logits, or probabilities;
- confusion counts for the important classes;
- a probability-sensitive score when confidence is part of the product;
- saturation, clipping, and non-finite counts.

A Brier score or log loss examines probability quality in a way that zero-one
accuracy cannot, but no single metric is universally sufficient. Choose metrics
that match the target functional and application cost. Keep the raw output
comparison so a stable task score cannot conceal an implementation regression.

### 16.5 Aggregate results must have slices and denominators

A global mean can pass while a rare class, long sequence, large shape, or
near-boundary group fails completely. Define required slices before viewing the
candidate result. Useful slices include:

- class or label of interest;
- sequence length, image size, batch size, or channel count;
- confidence or decision-margin bucket;
- padding ratio and valid-element count;
- stream position or chunk boundary;
- demographic or operational subgroup when relevant and lawful;
- backend, device, precision mode, and model version.

Report each metric's numerator, denominator, weighting, and missing-data policy.
For imbalanced classes, compare micro and macro summaries or explicit per-class
results. For variable-length data, a padded-element denominator can make a loss
look better merely by adding zeros; the valid mask and valid count belong to the
acceptance artifact.

### 16.6 Stateful systems need horizon and chunk tests

One-step agreement does not establish long-running equivalence. Recurrent
state, filters, normalization statistics, caches, and autoregressive generation
can accumulate small differences.

Use at least:

1. full-input versus correctly carried chunked execution;
2. deliberately reset or stale state as a mutation;
3. multiple chunk partitions, including one-element and uneven chunks;
4. short and long horizons;
5. save/resume at a supported boundary;
6. end-of-stream, flush, and reset behavior.

Record state outputs or checkpoints as first-class comparison targets when they
influence later results. A candidate that passes ten steps and drifts after one
thousand has not passed a one-thousand-step contract.

### 16.7 Match the gate to the change

| Change | Minimum acceptance |
|---|---|
| Algebraic rewrite or custom kernel | operator/intermediate, model output, and task replay |
| Compiler flags or vectorization | model output across shapes plus task replay; intermediates for sensitive reductions |
| Precision reduction or quantization | activation/output drift, saturation/calibration report, and task metric |
| Export or runtime/provider change | schema, complete outputs, dynamic envelope, placement, and task replay |
| Threshold-only product change | fixed model outputs plus threshold-specific utility and slice metrics |
| Stateful/chunking change | state tensors, chunk/full equivalence, horizon, reset, and task behavior |

Benchmark only candidates that pass the declared correctness and task gates.
Public performance suites follow the same general principle by defining a
model, dataset, and quality target rather than accepting throughput in isolation.
The exact quality floor remains application-specific; do not copy a benchmark's
percentage without its model, dataset, metric, and risk context.

### 16.8 Bounded acceptance experiment

A bounded CPU-only reference experiment for this guide used synthetic cases that make
both directions of failure visible. It verifies:

- a nominal three-level gate across dynamic batch sizes;
- a compensating internal error that preserves final output;
- argmax, binary-threshold, top-k, and greedy-token flips under small tensor
  differences;
- unchanged accuracy despite large logit drift;
- confidence degradation detected by a probability-sensitive score;
- aggregate accuracy hiding complete minority-slice failure;
- a padded-denominator false pass;
- full versus chunked state carry and a reset mutation;
- short-horizon tolerance followed by long-horizon drift;
- a bug visible only at a larger dynamic shape;
- batch-permutation metamorphism and an index-dependent mutation;
- clipping that preserves labels but destroys probability headroom.

This fixture does not establish production task thresholds, real dataset
coverage, stochastic decoding quality, accelerator/provider agreement, or
online drift monitoring. Those remain separate evidence obligations.

---

# Part III — Numerical behavior and precision

## 17. Numerical stability

### 17.1 Localize the first non-finite value

Do not report only “NaN occurred.” Record:

```text
phase
operator
tensor
index
batch or sequence position
minimum and maximum
first non-finite input
first non-finite output
build and backend
```

Insert checks after high-risk operations:

- exponential and logarithm;
- division and reciprocal square root;
- norms and variance;
- large reductions;
- gradient unscaling and clipping;
- optimizer moments;
- quantization scaling.

### 17.2 Stable reductions

A naive norm can overflow before clipping:

```c
sum += x * x;
```

Use a scaled sum-of-squares algorithm, a trusted library routine, or a wider accumulator when appropriate. `examples/safe_scaled_l2_norm.c` demonstrates a scalar scaled method.

Other useful techniques include:

- pairwise or tree reduction;
- compensated summation for sensitive sums;
- max-shifted log-sum-exp;
- online mean/variance algorithms;
- explicit FP32 accumulation for lower-precision inputs.

### 17.3 Conditioning versus implementation error

A large output difference may come from:

- an incorrect implementation;
- a poorly conditioned problem;
- a different but legal reduction order;
- a lower-precision algorithm;
- a different operator variant.

Use the fixture to distinguish them. Changing tolerance without understanding the scale is not a repair.

### 17.4 Fast-math and compiler transformations

Flags that allow reassociation or assume finite values can change:

- reduction order;
- signed-zero behavior;
- NaN/Inf checks;
- overflow and underflow behavior;
- reproducibility.

Use a strict floating-point build for reference and diagnosis. Enable aggressive math flags only in a separate performance configuration that must pass the differential suite.

## 18. Precision, mixed precision, fixed-point arithmetic, and quantization

### 18.1 Separate storage, compute, and accumulation precision

For every operator, write:

```text
input storage dtype
weight storage dtype
compute dtype
accumulator dtype
output dtype
master-parameter dtype
```

“FP16 operation” is ambiguous without this table.

### 18.2 A practical precision ladder

A useful progression is:

```text
double scalar reference
→ float scalar reference
→ optimized float backend
→ TF32/BF16/FP16 mixed path
→ quantized integer path
```

Each rung should be compared against the previous accepted reference with an explicit tolerance and task-level metric.

### 18.3 Mixed-precision training

Typical concerns include:

- operations that must remain in higher precision;
- loss scaling;
- gradient unscaling before clipping or inspection;
- overflow detection;
- master parameter copies;
- backend-specific use of tensor instructions;
- shapes that influence accelerator efficiency.

Automatic mixed precision is a policy engine, not a proof of numerical suitability. Inspect the actual dtypes and compare convergence and task metrics against a strict baseline.

### 18.4 Quantization as a DSP bridge

Experienced fixed-point developers already know many of the risks:

- dynamic range;
- saturation;
- accumulator width;
- rounding;
- per-stage scaling;
- representative calibration signals.

Modern affine quantization commonly expresses a real value approximately as:

```text
real ≈ scale × (integer - zero_point)
```

The additional AI-specific concerns are:

- per-tensor versus per-channel scale;
- symmetric versus asymmetric scheme;
- activation calibration dataset;
- observer/statistics policy;
- fake-quant behavior for quantization-aware training;
- unsupported operator fallback;
- model-level accuracy after graph conversion.

### 18.5 Quantized-kernel verification

Use at least three levels:

1. integer kernel versus a bit-exact integer reference where the specification permits it;
2. dequantized output versus float reference;
3. task-level metric versus the float model.

A bit-exact kernel can still implement the wrong model scale or tensor mapping. A good end metric can still hide a broken corner case.

CMSIS-NN is a useful example of optimized embedded kernels tied to a stated quantization contract and reference tests, but its supported dtypes and operators must be checked for the current version and target.

### 18.6 Make the affine integer contract executable

For a dense INT8 path, write the complete arithmetic rather than only saying
“quantized”:

```text
q = clip(round(real / scale) + zero_point)
acc[o] = bias_q[o] + Σ (x_q[i] - x_zero) (w_q[o,i] - w_zero[o])
bias_scale[o] = input_scale × weight_scale[o]
real_output[o] ≈ acc[o] × bias_scale[o]
```

Also declare integer ranges, rounding rule, accumulator width, saturation point,
scale granularity, and the axis used for per-channel parameters. Verify the
centered dot product against the algebraically expanded zero-point form, and
compare the native accumulator with a wider integer oracle. An INT8 input does
not imply that an INT16 or even INT32 reduction is always safe; prove the worst
reduction bound or reject overflow.

Calibration is a product decision. A wide min/max range may preserve an outlier
but waste resolution on common values; a clipped range may reduce ordinary
error while saturating rare values. Report both error and saturation counts,
then replay model outputs and task decisions. Do not accept a calibration policy
from tensor error alone.

One bounded CPU reference experiment built a strict C11 checked accumulator and obtained
**17/17 PASS**. Per-output-channel weights reduced MSE from about `0.00351` to
`0.00195`; dequantized row-argmax agreement with the float problem was `1.0`.
A clipped calibration range halved common-signal RMSE but produced five boundary
saturations, making the trade-off visible. INT16 wrap, wrong bias scaling, and
an INT32-overflow reduction were all detected.

Current LiteRT INT8 operator contracts commonly use asymmetric per-tensor
activations, symmetric weights with zero point 0, per-axis weight scales where
specified, and INT32 bias scaled by the product of input and weight scales.
CMSIS-NN follows the TensorFlow Lite for Microcontrollers quantization
specification. Treat that as an interoperability target, not proof: bit-exact
requantization, Cortex-M execution, cycles, memory, and energy still require the
actual current target library and toolchain.

## 19. Reproducibility, checkpointing, and tolerance

### 19.1 Define the reproducibility target

Possible targets differ:

- identical result in one scalar build;
- repeatable result on the same host/backend;
- tolerance-bounded result across CPU backends;
- stable training distribution across seeds;
- equivalent deployment metric across runtimes.

Do not demand bitwise identity where the backend contract does not provide it.

### 19.2 Reduction order and parallelism

Thread counts, vector ISA, GPU algorithm selection, and fused kernels can change floating-point order. Record them and test robustness, not only replay.

### 19.3 Checkpoint schema

Version the schema and record:

- parameter names, shapes, dtypes, and packing;
- optimizer and running state;
- step counters;
- RNG state;
- data cursor or sampler state;
- sequence/cache reset policy;
- endianness and format version;
- checksum or corruption detection.

Test:

- round trip;
- truncation and corruption;
- unsupported version;
- next-output replay;
- next-update replay.

Keep the schema next to the reader/writer code and include the format version in every round-trip fixture.

### 19.4 Tolerance policy

Choose tolerance by operation and scale. Record:

- absolute tolerance;
- relative tolerance;
- dtype;
- tensor magnitude;
- expected reduction depth;
- backend pair;
- whether NaN equality is legal.

One global tolerance for all tensors usually hides either false alarms or real defects.

---

# Part IV — Connect native numerical code to modern AI tools

## 20. Python and notebooks as a control and validation plane

Python is useful even when the production or hot-path kernel remains C/C++.

Use it for:

- generating tiny fixtures;
- calling an independent tensor library;
- finite differences and gradient checks;
- visualizing distributions and errors;
- preparing or inspecting model files;
- launching builds and tests;
- comparing CPU and GPU artifacts;
- documenting an investigation.

Jupyter notebooks combine live code, narrative, equations, and output. That makes them effective for exploratory validation and demonstrations. Their process state is also easy to forget: a displayed cell can depend on variables created by an earlier, later, or edited cell.

For any result that matters:

- restart the kernel and execute from top to bottom;
- keep reusable numerical logic in importable modules or native code rather than duplicated cells;
- write fixtures and outputs to ordinary files when another implementation must consume them;
- record the interpreter and environment used by the notebook kernel;
- reproduce the decisive check from a script, CLI, CTest, or CI job.

Do not make notebook execution order the only source of truth. A robust arrangement is:

```text
native library and CLI
+ deterministic fixture files
+ automated CTest/CI tests
+ notebook as a replaceable client and report
```

### 20.1 Binding choices

Common bridges include:

- Python C API;
- `ctypes` or CFFI for a stable C ABI;
- pybind11 for C++ bindings and NumPy buffer exchange;
- framework custom-operator APIs for full tensor-system integration.

A simple pointer bridge is enough for isolated experiments, but not always enough for autograd, compilation, export, device dispatch, or alias tracking.

### 20.2 Boundary checklist

At the language boundary, validate:

```text
shape
strides
dtype
item size
alignment
read/write permission
ownership and lifetime
device
contiguity
exception/error translation
GIL and thread behavior, if applicable
```

## 21. PyTorch as an executable oracle and integration host

PyTorch is useful in two distinct roles.

### 21.1 Reference role

Use a short eager implementation to produce:

- expected forward tensors;
- gradients;
- optimizer traces;
- randomized differential cases;
- dtype and device comparisons.

Keep the reference independent enough that it does not reproduce the same indexing code mechanically.

### 21.2 Operator-integration role

Use a registered custom operator when native C++/CUDA code must compose with framework features such as:

- autograd;
- `torch.compile`;
- export;
- device dispatch;
- fake/meta tensors;
- vectorization or tensor subclasses.

The operator is the public contract; the kernel is one implementation.

Current PyTorch guidance emphasizes:

- stable operator schema;
- accurate mutation and alias declarations;
- representative validation with `torch.library.opcheck`;
- fake/meta behavior matching real output metadata;
- explicit autograd registration when training is supported.

### 21.3 Integration ladder

A practical order is:

```text
standalone C/C++ fixture
→ Python binding to the standalone kernel
→ framework operator schema
→ CPU implementation
→ metadata/fake implementation
→ autograd registration or formula
→ CUDA implementation
→ opcheck and gradient tests
→ compile/export tests
```

Do not begin with every subsystem at once.

### 21.4 One operator creates several independent claims

A forward call returning the expected numbers proves only one layer. Treat the
following as separate claims with separate tests:

| Claim | Discriminating test | What it does not prove |
|---|---|---|
| Native arithmetic and bounds are correct | standalone unequal-shape fixture, sanitizer build, independent scalar reference | framework schema, gradients, export |
| The operator schema is truthful | mutation/alias tests and `torch.library.opcheck` | mathematical correctness |
| Fake or metadata execution matches the real kernel | compare shape, stride, dtype, device, layout, and storage behavior | element values |
| Autograd is registered in a supported way | `opcheck` with `requires_grad=True` inputs | correctness of the derivative formula |
| The derivative formula is correct | analytic comparison and `torch.autograd.gradcheck` in double precision | optimizer or long-run training behavior |
| The operator composes with compilation | a full-graph `torch.compile` test around surrounding framework operations | export or another runtime |
| The deployment path is correct | exported-model inspection and differential runtime fixtures | every execution provider or shape |

This separation prevents a common failure mode: using one successful eager
forward call as evidence for the entire integration stack.

### 21.5 A low-risk first bridge: versioned C ABI plus Python registration

A useful first integration seam for an existing C library is:

```text
versioned C ABI
→ ctypes/CFFI/pybind11 wrapper
→ functional custom-operator schema
→ fake kernel
→ registered autograd formula
```

For a functional operator such as `y = x*x + bias`, the native entry point can
remain independent of framework headers:

```c
#define AINP_NATIVE_ABI_VERSION 1u

unsigned int ainp_native_abi_version(void);
int ainp_square_plus_f64(
    const double *input,
    double *output,
    size_t element_count,
    double bias);
```

The wrapper must reject unsupported dtype, device, and layout before passing
raw pointers. The operator then declares no mutation and returns fresh storage:

```python
@torch.library.custom_op(
    "ainp_native::square_plus",
    mutates_args=(),
    device_types="cpu",
)
def square_plus(x: torch.Tensor, bias: float) -> torch.Tensor:
    require_cpu_float64_contiguous(x)
    out = torch.empty_like(x)
    call_versioned_c_abi(x.data_ptr(), out.data_ptr(), x.numel(), bias)
    return out

@square_plus.register_fake
def _(x, bias):
    require_float64_contiguous_metadata(x)
    return torch.empty_like(x)
```

The fake implementation must describe the real result rather than merely
returning a tensor with the right dimensions. Strides, dtype, device, layout,
and storage behavior are part of the contract. It must not inspect real data.

Add training support separately:

```python
def setup_context(ctx, inputs, output):
    x, _bias = inputs
    ctx.save_for_backward(x)

def backward(ctx, grad_output):
    (x,) = ctx.saved_tensors
    return grad_output * (2.0 * x), None

torch.library.register_autograd(
    square_plus, backward, setup_context=setup_context
)
```

Write the backward formula using framework-visible operations or other
registered operators. Do not call an opaque NumPy or native routine directly
from the formula unless it is itself exposed through a supported operator
boundary.

### 21.6 What `opcheck` proves, and what it does not

Run `torch.library.opcheck` on representative valid inputs, including empty
shapes, unequal dimensions, supported dtypes, and training inputs when
applicable. Its role is integration correctness: schema, mutation/alias
behavior, fake tensors, autograd registration, and compiler-facing dispatch.

It is deliberately not the numerical oracle. Pair it with:

```text
standalone native reference comparison
+ eager framework expression comparison
+ double-precision gradcheck
+ invalid-input rejection tests
+ compiled composition test
```

A good negative test is to deliberately return incorrect fake strides and
confirm that the integration test rejects the operator. The current PyTorch
2.13 tutorial demonstrates this check, but older releases may not reject the
same defect. Therefore record the framework version and keep direct assertions
for shape, stride, dtype, device, layout, and aliasing even when `opcheck`
passes. A suite that never observes metadata is too weak for framework
integration.

### 21.7 Choose the ABI boundary deliberately

PyTorch's Stable C++ API is intended for ahead-of-time compiled extensions and
precompiled custom-operator packages that must remain binary-compatible across
PyTorch versions. Use it when its supported operator and utility surface is
sufficient. Use the regular ATen/LibTorch API when required functionality is not
available through the stable API, accepting the corresponding version coupling
and build matrix. Record the minimum compatible version of every stable symbol
used; the surface is stable, but individual APIs have explicit introduction
versions.

A plain C ABI remains useful when:

- the kernel already belongs to a framework-independent native library;
- the first goal is CPU validation rather than direct ATen tensor access;
- allocator ownership and error results can be kept on one side of the ABI;
- a narrow wrapper is easier to test than a framework-coupled rebuild.

The C ABI route is not automatically better. It adds wrapper validation and may
require explicit copies or contiguous-only policies. Record that cost instead
of hiding it.

### 21.8 Verify both ABI axes with the built wheel

"Stable ABI" can refer to two independent compatibility boundaries. Do not
collapse them into one claim.

| Boundary | Build control | Artifact evidence | Required compatibility test |
|---|---|---|---|
| CPython limited API | `py_limited_api`, `Py_LIMITED_API`, and an `abi3` wheel tag | the extension imports without ordinary CPython-version-specific bindings | install the same wheel on every supported Python minor version |
| LibTorch stable ABI | `torch::stable` APIs and a minimum `TORCH_TARGET_VERSION` | the binary avoids `libtorch_python` and uses only the supported stable surface | install the same wheel, without rebuilding it, against every claimed PyTorch release |

The wheel name is not proof. A useful acceptance sequence is:

```text
build the wheel from a clean tree
→ record the compiler-defined limited-API version and wheel tag
→ unpack the wheel and inspect native dependencies
→ install into an empty directory outside the source tree
→ import and run forward, metadata, autograd, and compile tests
→ repeat with the identical wheel across the claimed Python/PyTorch matrix
```

Do not copy an `abi3` tag from a newer tutorial while an older build tool injects
a different `Py_LIMITED_API` value. Derive or inspect the actual build value and
make the filename, metadata, and compiler flags agree. Likewise, a wheel that
works with the PyTorch version used to build it has not yet proved LibTorch
version independence.

A clean consumer test should also verify that:

- the package path points to the installed wheel, not the source directory;
- the operator returns the declared shape, stride, dtype, device, and aliasing;
- unsupported dtype, device, and layout fail explicitly;
- the shared library does not depend on `libtorch_python` when limited-API
  packaging is claimed;
- the same package composes with `opcheck`, registered autograd, and at least one
  full-graph compiler test relevant to the deployment path.

### 21.9 Windows DLL boundaries: prove four contracts separately

A Windows DLL is not merely a Linux shared library with another suffix. Treat
these as four independent acceptance claims:

| Contract | Questions | Minimum evidence |
|---|---|---|
| Machine and loader | x64, x86, or ARM64? Which process loads which DLL, and from what directory? | `dumpbin /headers`, a clean directory, and an explicit loader/search-path test |
| Symbol and call ABI | Which names are exported? C or C++ linkage? Which calling convention? Which ABI version? | `extern "C"`, explicit export/import macros, `dumpbin /exports`, and consumer import inspection |
| Runtime and ownership | Who allocates, frees, aligns, mutates, retains, and reports errors? Which `/MD` or `/MT` policy applies? | an ownership table, same-side allocation/free, negative tests, and dependency inspection |
| Toolset and deployment | Which compiler, linker, Windows SDK, redistributable, and dependent DLL versions are required? | recorded tool versions, `dumpbin /dependents`, and a clean consumer without developer paths |

Current Microsoft documentation describes broad binary compatibility among
MSVC v14x toolsets from Visual Studio 2015 onward, with restrictions: the linker
must be at least as new as the newest input, and the installed v14
Redistributable must be at least as new as the build tools. That compatibility
statement is useful, but it does not prove that an arbitrary exported C++ class,
STL object, exception, allocator, or framework object is a safe public ABI.
Prefer a narrow versioned C boundary unless the complete producer/consumer build
contract is controlled.

A small explicit header is easier to inspect than relying on automatic export of
every visible function:

```c
#if defined(_WIN32)
  #if defined(AINP_BUILD_DLL)
    #define AINP_API __declspec(dllexport)
  #else
    #define AINP_API __declspec(dllimport)
  #endif
  #define AINP_CALL __cdecl
#else
  #define AINP_API
  #define AINP_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define AINP_ABI_VERSION 1u
AINP_API unsigned int AINP_CALL ainp_abi_version(void);
AINP_API int AINP_CALL ainp_run_f32(
    const float* input, float* output,
    unsigned long long count, float bias);

#ifdef __cplusplus
}
#endif
```

For ownership, choose one of these patterns and write it into the API:

1. **Caller-owned buffers:** the caller allocates and frees; the DLL validates
   size, alignment, overlap, and lifetime and does not retain the pointer.
2. **Opaque handle with same-side destruction:** the DLL creates an opaque
   handle and exports the matching destroy function.
3. **Explicit allocator interface:** only when necessary, pass versioned
   allocate/free callbacks and define thread-safety, alignment, failure, and
   lifetime rules.

Do not allocate with `new`, `malloc`, or `_aligned_malloc` in one component and
silently free with another component's `delete`, `free`, or `_aligned_free`.
Microsoft specifically requires `_aligned_free` for `_aligned_malloc`, and its
CRT guidance warns about memory and CRT objects crossing DLL boundaries. Even
when both components use a compatible dynamic CRT, same-side ownership remains
the simpler public contract. Avoid passing `std::string`, `std::vector`, `FILE*`,
locales, exceptions, or other CRT/STL-owned objects through a long-lived plugin
ABI.

Make the runtime choice visible in CMake rather than allowing project flags to
drift:

```cmake
cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0091 NEW)
project(native_boundary LANGUAGES C CXX)

add_library(native_boundary SHARED native_boundary.c)
set_property(TARGET native_boundary PROPERTY
  MSVC_RUNTIME_LIBRARY
  "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")  # /MD or /MDd
```

Use the same explicit policy for every target that exchanges CRT-owned objects,
but do not use a matching `/MD` choice as a substitute for an ownership
contract. A library that never transfers allocator-owned objects can often keep
the boundary simpler.

A Windows clean-consumer gate should be mechanical:

```text
build a Release or RelWithDebInfo x64 DLL and import library
→ inspect headers, exports, imports, and dependent DLLs
→ copy only the consumer, target DLL, and declared runtime dependencies
→ remove source/build directories and scrub developer-only PATH entries
→ run valid, invalid, alignment, aliasing, and ABI-version cases
→ rename or remove the target DLL and require loader failure
→ repeat for every claimed architecture, CRT policy, and supported toolset
```

Use a fully qualified path or a deliberately constrained DLL search policy for
plugins. The Windows loader search order is part of security and correctness;
a developer `PATH` that happens to contain the right dependency is not deployment
evidence.

One bounded structural reference experiment cross-compiled a CRT-free x64 PE/COFF DLL and
consumer with `clang-cl` 17 targeting the MSVC ABI and `lld-link` 17. Structural
inspection passed six checks: PE32+ x64 identity, exact three-symbol export and
import contracts, the expected DLL import name, and no DLL import directory for
the deliberately dependency-free library. This is structural evidence only.
Native Windows execution, Microsoft `cl.exe`, `dumpbin`, loader behavior,
Redistributable deployment, `/MD` versus `/MT`, MSVC AddressSanitizer, x86, and
ARM64 remain **NOT_RUN**.

## 22. ONNX and ONNX Runtime as exchange and deployment boundaries

ONNX provides versioned operator schemas and optional shape/type inference. ONNX Runtime provides cross-platform execution providers and native extension points.

Use ONNX when the goal is:

- framework-neutral model exchange;
- deployment validation;
- operator/version inspection;
- comparing a native runtime against a training framework;
- integrating a hardware-specific execution provider or custom operator.

### 22.1 Treat export as another implementation

Export can change or decompose operations. Verify:

- operator set version;
- input and output names;
- dynamic axes and shape inference;
- dtype conversion;
- constant folding;
- unsupported or custom operators;
- preprocessing and postprocessing outside the graph;
- numerical output on fixed fixtures.

### 22.2 Custom operators

ONNX Runtime custom operators can be packaged in a shared library. Define:

- domain and version;
- input/output types and shapes;
- provider/device support;
- threading and state behavior;
- serialization expectations;
- fallback behavior.

Do not use a custom op to conceal an undefined contract.

A shared custom-operator library is part of the deployable artifact. Current
ONNX Runtime documentation requires a separately loaded custom-op library to
export `RegisterCustomOps`, which receives session options and a versioned C API
base. The runtime-managed registration API loads the library and keeps its
handle alive while sessions still reference it.

That C entry point is only one compatibility layer. Also record and test:

- the ONNX Runtime API version requested by the headers;
- exported symbol spelling and calling convention;
- C++ runtime, CRT, standard-library, and other native dependencies;
- DLL/SO discovery and transitive search paths;
- allocator and ownership rules across the boundary;
- provider, device, architecture, and compiler assumptions.

A plugin can use the loader-supplied C API without directly linking to
`libonnxruntime`; this reduces one deployment dependency, but it does not make
the produced DLL or SO universally portable. Test the application with a clean
runtime installation and no developer build tree in the import or library
search path.

### 22.3 Export and runtime differential gate

A PyTorch custom operator does not automatically become an ONNX operator. An
export path needs one of the following:

- a decomposition into supported standard ONNX operators;
- an exporter translation for the PyTorch operator;
- a custom ONNX-domain node plus a matching runtime kernel.

Test export and execution as two separate stages:

```text
1. export succeeds for fixed and declared dynamic shapes
2. inspect domain, operator name, version, types, and shape information
3. load with a clean ONNX Runtime session
4. assert the requested execution providers and their precedence
5. run the same deterministic fixtures in eager PyTorch and ONNX Runtime
6. compare outputs with a scale-aware tolerance
7. inspect provider placement and unexpected CPU fallback
8. repeat with I/O binding when transfer cost is part of the claim
```

Provider order is a runtime decision. Missing provider kernels may cause CPU
fallback, and host/device copies can dominate a short graph. Verbose placement
logs and I/O binding are therefore evidence, not optional tuning folklore.

For a custom runtime operator, also test:

```text
unregistered model        → expected session-creation failure
missing or wrong library  → expected registration/load failure
wrong domain/version      → expected session-creation failure
binary contract           → exported entry point and inspected dependencies
valid library             → clean registration, load, and deterministic output
unsupported dtype/shape   → explicit error, not silent reinterpretation
```

### 22.4 Run a deployment fixture, not one exporter example

An exporter success report is useful, but it normally exercises the supplied
example inputs. It does not cover every declared dynamic shape, value range,
runtime provider, or ownership path. Build a small deployment fixture with
separate gates:

| Gate | Minimum evidence |
|---|---|
| Export | no fallback, saved exporter report, fixed input/output names, explicit opset |
| Structure | ONNX checker, shape/type inference, domain/op/version inspection |
| Dynamic envelope | several unequal, non-square shapes plus boundary sizes |
| Runtime | an explicit provider list and the provider order returned by the session |
| Numerical comparison | eager-framework versus runtime outputs on deterministic cases |
| Placement | runtime profiling or placement logs showing the provider for each node |
| Ownership | ordinary `run` plus I/O Binding with caller-owned or device-resident buffers |
| Negative behavior | wrong dtype/rank, unsupported broadcast, missing custom domain/library |
| Native library | exported registration symbol, architecture, direct and transitive dependencies, clean search path |
| Artifact | serialize and reload any optimized model in the environment for which it was produced |

For a tiny graph such as `y = x*x + bias`, useful cases include `[2,3]`, `[1,5]`,
`[4,2]`, and `[3,7]`. Square-only examples are weak because they hide exchanged
dimensions and some broadcasting mistakes.

Provider availability, session provider order, and actual node placement are
three different observations. A requested accelerator can be available while
some nodes still execute on CPU. Record placement from the runtime rather than
inferring it from the constructor argument.

### 22.5 Use I/O Binding as an ownership and transfer test

I/O Binding is not merely a performance option. It makes buffer placement and
ownership visible. On CPU, a preallocated-output test can prove that the runtime
writes into caller-provided storage:

```python
out = np.full(x.shape, np.nan, dtype=np.float64)
io = session.io_binding()
io.bind_ortvalue_input("x", ort.OrtValue.ortvalue_from_numpy(x, "cpu"))
io.bind_ortvalue_input("bias", ort.OrtValue.ortvalue_from_numpy(bias, "cpu"))
io.bind_ortvalue_output("y", ort.OrtValue.ortvalue_from_numpy(out, "cpu"))
session.run_with_iobinding(io)
io.synchronize_outputs()
assert not np.isnan(out).any()
```

On an accelerator, extend the same contract with device identity, stream or
queue synchronization, pointer lifetime, and whether a host/device copy occurs.
Do not claim zero-copy from API usage alone; measure or inspect the actual path.

### 22.6 Treat registration and session creation as separate gates

Create a tiny model containing the intended custom domain and operator, then try
to construct a session without the custom library. Expected failure before the
first run proves that the runtime is not silently substituting a standard op.
Separately try to register a nonexistent or wrong library. Finally, repeat from
a clean installation with the valid library and with a deliberately wrong
domain or version.

The clean consumer should receive only the artifacts that production needs:
for example, the model, the custom-op library, and a small runner. Clear source
tree paths, user-site imports, and developer library paths. Build or install the
runtime independently, then inspect the loaded plugin's exported entry point and
native dependencies before accepting the numerical result.

One bounded Linux x86-64 CPU reference experiment used a fresh virtual environment with
ONNX Runtime 1.27.0. It built a C++ shared library exporting
`RegisterCustomOps`, verified that the ELF binary did not directly depend on
`libonnxruntime`, copied only the model, library, and runner into a temporary
consumer directory, and obtained **17/17 PASS** across the standard-operator and
custom-operator gates. The custom model failed before registration and with a
missing library; after registration, five dynamic non-square float64 cases were
bit-exact and three invalid-input cases were rejected. This is evidence for that
Linux CPU artifact, not evidence for Windows, macOS, mobile, or accelerator
builds.

## 23. CPU path: scalar, compiler, BLAS, and deep-learning libraries

### 23.1 Keep a readable scalar reference

The scalar path should prioritize:

- obvious indexing;
- named dimensions;
- checked bounds in debug mode;
- diagnostic dumps;
- no layout trick that obscures the equation.

### 23.2 Let the compiler show what it can do

Before writing intrinsics:

- remove unnecessary aliasing;
- make loop bounds clear;
- align where legitimately guaranteed;
- keep the innermost access contiguous;
- inspect vectorization remarks;
- benchmark representative shapes.

The fastest manual SIMD version is not automatically the best maintenance reference.

### 23.3 BLAS mapping

BLAS is a strong target when the work maps cleanly to GEMM/GEMV and the shapes are large enough to amortize call and packing overhead.

Record:

```text
row/column convention
M, N, K
transpose flags
leading dimensions
alpha and beta
batch mapping
thread count
packing/reorder time
```

Small, skinny, or repeatedly repacked matrices may require a different strategy.

### 23.4 oneDNN and backend-selected formats

Deep-learning libraries may choose blocked or opaque formats for performance. oneDNN explicitly models primitives, engines, streams, and memory descriptors; format propagation is a central performance concept.

Treat backend-selected layout as part of the execution plan. Reorders between individually fast primitives can dominate the graph.

### 23.5 Threading

Control and record:

- application thread count;
- framework inter-op thread count;
- framework intra-op thread count;
- BLAS/OpenMP/backend thread count;
- data-loader or process-worker count;
- nested parallelism;
- affinity and NUMA placement;
- reduction order;
- oversubscription.

PyTorch exposes intra-op and inter-op controls, and thread configuration should be applied before the relevant eager, autograd, or compiled work begins. ONNX Runtime likewise exposes session-level thread controls. These settings can interact with OpenMP, MKL, oneDNN, and application-level pools; do not tune any one of them in isolation.

A useful CPU matrix is:

```text
1 physical thread
1 process × N intra-op threads
N processes × 1 intra-op thread
selected intermediate combinations
SMT off/on when controllable
NUMA-local versus cross-NUMA placement
```

Measure both throughput and tail latency. A single-thread scalar fixture and a threaded production path answer different questions.

### 23.6 Carry one accepted operation through every candidate path

Do not compare unrelated demos. Hold one already accepted operation constant and
change only the implementation path. A useful dense fixture is:

```text
C[M,N] = A[M,K] × B[K,N] + bias[N]
```

For each path, retain the same inputs and report:

```text
implementation identity and build flags
shape, dtype, layout, transpose flags, and thread budget
correctness against the same oracle
allocation, packing, reorder, and bias/fusion scope
warm-up and sample policy
median and range or tail statistic
first shape at which the ranking changes
```

The name of an API is not the identity of its implementation. `cblas_sgemm`
may resolve to a reference library, OpenBLAS, BLIS, MKL, Accelerate, or another
provider. Likewise, a framework call may dispatch differently by shape, dtype,
layout, build, and CPU. Record the linked library or build configuration; use
backend diagnostics when available. oneDNN verbose output is one example of
runtime evidence for primitive selection and reorders.

Compiler output is another implementation path, not a presumed baseline.
Retain a vectorization report or inspect the generated code. `-O3` alone does
not prove that the intended inner loop vectorized, nor that the resulting loop
is faster for the tested shape.

### 23.7 Read a crossover table, not a backend leaderboard

One bounded single-thread Linux x86-64 reference experiment compared a scalar double oracle,
scalar float C++, GCC-vectorized float C++, system CBLAS, NumPy/OpenBLAS, and a
PyTorch CPU path. Inputs were contiguous and prepared, output storage was
reused, allocation was excluded, output overwrite and bias addition were
included, and each result passed a declared float32 tolerance plus row-argmax
agreement.

| M × K × N | Fastest recorded path | Median per call | Second relevant path |
|---|---|---:|---|
| 1 × 64 × 16 | compiler-vectorized C++ | 0.187 µs | scalar float: 0.725 µs |
| 8 × 64 × 32 | compiler-vectorized C++ | 1.995 µs | NumPy/OpenBLAS: 3.473 µs |
| 32 × 128 × 64 | PyTorch CPU | 9.075 µs | NumPy/OpenBLAS: 9.162 µs |
| 128 × 256 × 128 | PyTorch CPU | 83.249 µs | NumPy/OpenBLAS: 103.659 µs |
| 256 × 256 × 256 | PyTorch CPU | 292.982 µs | NumPy/OpenBLAS: 367.181 µs |

This table demonstrates a method, not a portable ranking. The direct generated
loop won the two smallest cases, while optimized library/framework paths won
from the middle case onward. The system CBLAS path resolved to the system
reference-style BLAS and was not an optimized vendor implementation; the BLAS
function name alone therefore predicted nothing about speed. At tiny sizes,
call and control overhead were part of the observed boundary. At larger sizes,
the optimized kernels amortized that boundary.

Do not generalize the exact crossover without repeating the matrix on the
target machine. CPU frequency and affinity, multithreaded scaling, NUMA,
direct oneDNN primitive and reorder cost, other compilers, and other
architectures were not measured in that fixture.

Use the result to make a scoped decision:

- retain the scalar double path as a numerical oracle, not a performance target;
- keep a simple generated loop when the production shape is tiny and stable;
- use BLAS only after identifying the actual provider and exact mapping;
- prefer a framework or primitive library when its measured range wins and its
  packing, thread, and integration costs are included;
- dispatch by shape only when the extra branches, tests, and maintenance are
  justified by an end-to-end gain.

## 24. GPU path: a different execution and memory model

GPU work is not “SIMD C with more lanes.” The important concepts are execution hierarchy, memory movement, and asynchronous scheduling.

### 24.1 Minimal mental model

For CUDA-style programming:

- a kernel launch creates a grid of thread blocks;
- threads execute in warps;
- blocks can cooperate through shared memory and synchronization;
- global-memory access patterns influence transaction count;
- host launches and device execution are normally asynchronous.

### 24.2 Port in layers

A useful progression is:

```text
CPU scalar reference
→ simple correct GPU kernel
→ CPU/GPU differential fixture
→ transfer and synchronization audit
→ memory-access optimization
→ occupancy and launch-shape tuning
→ fusion or vendor-library substitution
→ mixed precision
```

### 24.3 Time the correct scope

Separate:

- context and module initialization;
- allocation;
- host-to-device transfer;
- kernel launch overhead;
- kernel execution;
- synchronization;
- device-to-host transfer;
- steady-state pipeline time.

A CPU timer around an asynchronous launch does not measure kernel completion unless a valid synchronization or event measurement is used.

### 24.4 Memory access

Check:

- coalescing of global access;
- strided access;
- shared-memory bank behavior;
- register pressure and spills;
- temporary workspace;
- data reuse;
- alignment and vectorized loads;
- host/device transfer overlap.

NVIDIA’s CUDA Best Practices Guide and Nsight tools are primary references for current hardware behavior and measurement.

### 24.5 Profile at two levels

Use a system timeline to answer:

- Is the GPU idle?
- Are transfers serialized?
- Is launch overhead dominant?
- Are CPU threads or data loading the bottleneck?

Use a kernel profiler to answer:

- Is the kernel limited by memory, instructions, occupancy, or dependencies?
- Are accesses coalesced?
- Are expected tensor instructions used?
- Is there excessive divergence or spilling?

Nsight Systems and Nsight Compute serve these different roles.

### 24.6 Vendor libraries first when the mapping is standard

Use cuBLAS, cuDNN, or another maintained backend when the operation maps well and the dependency is acceptable. Write a custom kernel when there is a demonstrated reason:

- unusual fusion;
- unsupported layout or operator;
- very small or specialized shapes;
- stateful streaming behavior;
- hardware research;
- measured backend overhead.

A custom kernel inherits testing, portability, precision, and maintenance obligations.

### 24.7 Choose the GPU programming level deliberately

A practical order is:

```text
framework-native operator
→ vendor library
→ framework compiler/fusion
→ GPU kernel DSL such as Triton
→ CUDA, HIP, or another low-level vendor path
```

A kernel DSL can shorten iteration for elementwise, reduction, normalization, and fusion work while still requiring reference comparison and representative benchmarking. CUDA provides the deepest NVIDIA-specific control. HIP provides a CUDA-aligned C++ runtime and kernel language for AMD-oriented and cross-vendor work, but source similarity does not remove architecture-specific tuning.

Choose based on:

- target hardware and deployment support;
- required language/runtime integration;
- operator semantics and fusion opportunity;
- debugging and profiling support;
- maintenance ownership;
- measured gap after standard backends;
- portability boundary actually required.

Do not introduce a custom GPU language merely to reproduce a library operation at lower reliability.

### 24.8 Model overlap as a bounded ownership state machine

A double-buffered pipeline is correct only if each buffer has an explicit state:

```text
FREE → FILLING → READY → COPYING → IN_FLIGHT → FREE
```

Associate each transition with an owner and a completion condition. A host buffer
must not return to `FREE` until the asynchronous copy that reads it is complete.
A device output must not be read by the CPU until the device-to-host copy or
producer event is complete. Generation tags help reject stale releases when ring
slots are reused.

Use a bounded queue so overload creates backpressure rather than unbounded
allocation. Benchmark at least:

- transfer alone;
- compute alone;
- serialized transfer plus compute;
- steady-state overlapped throughput;
- end-to-end latency including fill, drain, preprocessing, and synchronization.

A timeline should show the expected overlap. A faster loop without a timeline or
completion test may only have moved synchronization outside the timed region.

---

# Part V — Build, test, diagnose, and measure

## 25. Build and test ecosystem

### 25.1 Suggested build matrix

At minimum:

| Configuration | Purpose |
|---|---|
| `debug` | assertions, symbols, checked views |
| `asan-ubsan` | address and undefined-behavior checks |
| `msan` where supported | uninitialized-value checks |
| `tsan` for threaded paths | data-race checks |
| `strict-release` | optimized without aggressive floating-point assumptions |
| `performance` | target ISA/backend and approved math flags |
| `cuda-debug` | device assertions, line info, conservative optimization |
| `cuda-performance` | production GPU configuration |

Sanitizers cover different classes and are not generally interchangeable.

### 25.2 Treat Windows as an explicit build matrix

Do not label one successful Visual Studio build as "Windows support." Record the
axes that can change binary behavior:

| Axis | Examples | Why it matters |
|---|---|---|
| Architecture | x64, x86, ARM64 | pointer width, calling convention details, instruction set, and dependency architecture must match |
| Toolset and linker | supported v14x toolset and exact linker version | the linker must be new enough for every input binary |
| Windows SDK | exact installed/selected SDK | headers, import libraries, and minimum target APIs can differ |
| CRT | `/MD`, `/MDd`, `/MT`, `/MTd` | dependency deployment and CRT-owned state/objects differ |
| Configuration | Debug, RelWithDebInfo, Release | assertions, optimization, debug runtime, and redistributability differ |
| Instrumentation | ordinary, `/fsanitize=address`, profiling | instrumentation changes runtime dependencies and incompatible flags |
| Consumer origin | developer tree, clean directory, clean VM/runner | accidental `PATH`, SDK, or build-tree dependencies become visible |

Use `dumpbin` or an equivalent PE/COFF inspector to retain headers, exports,
imports, and dependencies as CI artifacts. Run the consumer outside the source
and build directories. For deployable evidence, prefer Release or
RelWithDebInfo; a Debug executable may require debug runtime components that are
not a production deployment contract.

MSVC AddressSanitizer is a useful separate configuration, not a replacement for
ordinary release testing. Current Microsoft documentation enables it with
`/fsanitize=address` on supported x86/x64 configurations and notes conflicts
with options such as `/RTC` and incremental linking. Keep the sanitizer preset
separate so the ordinary ABI and deployment artifact is still tested.

### 25.3 CMake Presets and CTest

CMake Presets provide shareable configure/build/test settings; `CMakeUserPresets.json` can hold local-only choices. Use CTest so fixtures run outside a notebook and under CI.

A useful command surface is:

```text
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

The exact preset names are project decisions. The value is that the tested configuration becomes a checked-in artifact rather than a remembered command line.

### 25.4 Warnings and assertions

Enable strong warnings appropriate to the compiler, but distinguish:

- compiler diagnostics;
- runtime assertions;
- sanitizers;
- numerical invariants;
- framework operator checks.

Each observes a different failure class.

### 25.5 CI artifacts

For failures, retain compact artifacts such as:

- exact command and environment summary;
- failing seed and fixture;
- first divergent tensor;
- sanitizer log;
- benchmark JSON;
- profiler range or report identifier;
- source revision and diff.

Do not rely only on a screenshot of final loss.

## 26. Benchmark and profiling method

### 26.1 Correctness gate before timing

A backend enters performance comparison only after it passes the declared fixture and tolerance suite.

### 26.2 Use a shape matrix

AI performance is shape-dependent. Include:

- tiny diagnostic shapes;
- realistic common shapes;
- awkward non-multiples and tails;
- batch 1 and throughput-oriented batches;
- small and large reduction dimensions;
- contiguous and accepted non-contiguous layouts;
- precision variants.

### 26.3 Separate setup and steady state

Report separately:

- model or primitive creation;
- allocation;
- packing/reorder;
- compilation/autotuning;
- warm-up;
- steady-state operation;
- transfer and synchronization;
- end-to-end request or training step.

### 26.4 Arithmetic intensity and memory traffic

Many AI operators are GEMM-like, but small elementwise operations and reorders can be bandwidth or launch limited. Ask:

```text
How many useful operations are performed per byte moved?
How often is the same value reused?
Is a separate pass avoidable through fusion?
Is packing amortized?
```

Use measurement rather than assuming every neural-network operation is compute-bound.

### 26.5 Report distributions

Use multiple iterations and report at least:

- median;
- a tail percentile or range;
- warm-up policy;
- clock/power state when relevant;
- thread/stream count;
- work units such as elements, tokens, frames, or operations per second.

### 26.6 Validate profiler interpretation

Profilers have overhead and collection limits. Use focused ranges, confirm that the measured workload is representative, and retain a simple external timing sanity check.

## 27. Diagnostic playbook

| Symptom | Likely layers | First discriminating test |
|---|---|---|
| Correct only for square shapes | tensor/kernel | unequal dimensions with diagnostic values |
| Correct for contiguous input only | tensor/binding | explicit stride fixture and rejection test |
| Forward agrees; backward differs | gradient/graph | finite differences and per-route accumulation dump |
| Difference is a constant factor | reduction/loss | print numerator, denominator, and update scaling |
| NaN begins in norm or variance | numerical | scaled reduction and first-non-finite trace |
| CPU agrees; GPU differs slightly | precision/reduction | strict FP32 case, then dtype/algorithm audit |
| GPU result is stale or intermittent | runtime | explicit error check and synchronization at fixture boundary |
| GPU is slow for tiny work | runtime/performance | separate launch and transfer from kernel time |
| INT8 output clips heavily | quantization | saturation histogram and calibration-range audit |
| Notebook result cannot be reproduced | orchestration/build | fresh process, scripted fixture, recorded environment |
| Exported model differs | graph/export | compare named intermediate outputs and opset mapping |
| Resume diverges immediately | checkpoint/state | next-output and next-update replay fixture |
| Optimized release differs from debug | compiler/numerical/UB | strict release, sanitizers, then fast-math/alias audit |

### Debug one layer at a time

Avoid simultaneously changing:

- layout;
- precision;
- backend;
- batching;
- optimizer;
- threading;
- compiler flags.

A good repair cycle is:

```text
freeze failing fixture
→ identify earliest divergence
→ change one contract or implementation layer
→ rerun local fixture
→ add regression test
→ rerun broader matrix
```

Preserve negative evidence. Knowing that a hypothesis failed prevents the next developer or agent from repeating the same branch.

## 28. Completion criteria and defensible claims

### 28.1 Minimum numerical-component completion

A component is ready for broader integration when:

- the six contracts are documented;
- tiny unequal-shape fixtures pass;
- edge and invalid-input behavior is defined;
- reference and production paths agree within policy;
- backward and state traces pass when applicable;
- accepted layouts and dtypes are tested;
- preprocessing identity, collation ownership, lengths, masks, padding, and valid-element denominators are tested when data is variable-length;
- bounded buffers, completion events, and source-reuse rules are tested when the pipeline is asynchronous;
- covered sanitizer/build configurations pass;
- failure output identifies the first useful divergence.

### 28.2 Framework-integration completion

Additionally:

- schema and mutation/aliasing behavior are accurate;
- metadata/fake behavior matches real output;
- device dispatch is tested;
- autograd is registered and checked when supported;
- compile/export behavior is tested or explicitly unsupported;
- ABI and packaging requirements are recorded;
- each claimed Windows artifact has inspected architecture, exports, imports,
  CRT/dependent-DLL policy, loader path, and a clean-consumer result.

### 28.3 Optimization completion

Additionally:

- the bottleneck was measured;
- setup and steady-state time are separated;
- optimized paths pass the same fixtures;
- layout and conversion costs are included;
- precision trade-offs are measured at tensor and task level;
- hardware, software, shapes, and variance are reported.

### 28.4 Claim wording

Prefer bounded wording:

- “Matches the double scalar fixture for the tested shapes within the stated tolerance.”
- “Passes analytic, finite-difference, and independent-autodiff gradient checks on the fixture.”
- “No ASan/UBSan failure was observed on the covered test matrix.”
- “The CUDA path is faster for the reported shapes when transfer and warm-up are excluded/included as stated.”
- “INT8 inference preserves the reported task metric within the stated calibration and dataset conditions.”

Avoid unqualified claims such as “correct,” “deterministic,” “GPU accelerated,” or “production ready” without saying what evidence and boundary support them.

---

# Case studies

The guide uses a small set of recurring examples rather than requiring a separate public repository:

1. a dense layer followed by stable softmax cross-entropy, used for forward, backward, and finite-difference checks;
2. a quantized dot product, used to connect fixed-point/DSP practice to modern quantization contracts;
3. a stateful recurrent cell, used to expose state lifetime, packed parameters, and backward-through-time issues;
4. a custom-operator boundary, used to connect verified native code to framework dispatch and export;
5. a CPU–GPU pipeline, used to expose transfer, launch, synchronization, and overlap;
6. a pre-norm causal attention block with dynamic/static KV caches, used to integrate masks, stable softmax, precision, ownership, and chunk/full state equivalence.

LSTM-specific advice is one instance of the broader contracts, not the organizing principle of the guide.

---

# Appendix A — Assisted audits and recoverable work

An LLM or coding agent can help with:

- enumerating parameter and tensor paths;
- generating unequal-shape fixtures;
- writing independent reference code;
- comparing dumps and locating the first divergence;
- searching current backend documentation and issues;
- preparing build/test matrices;
- reviewing final diffs.

Do not treat generated prose or a successful tool call as completion evidence.

A recoverable handoff should include:

```text
objective
canonical guide and exact code revision
active contract
raw observation
supported interpretation
open hypothesis
completed commands and outputs
NOT_RUN checks
next smallest discriminating action
```

Keep this structure in ordinary files. Human-readable, explicit, testable documentation is naturally reusable by ChatGPT and agentic tools without making the public guide machine-first.

The handoff fields above are intentionally compact enough to paste into an issue, experiment log, or ordinary Markdown file.

# Appendix B — Compact checklist

Before trusting or optimizing a native AI numerical component:

```text
[ ] Equation and reduction policy written
[ ] Shape, strides, dtype, device, and layout written
[ ] Mutation, aliasing, ownership, and lifetime written
[ ] Preprocessing identity and sample IDs recorded
[ ] Length, padding, mask polarity, and valid denominator written
[ ] Async buffers have bounded capacity, completion, and reuse rules
[ ] Unequal-dimension deterministic fixture exists
[ ] First-intermediate comparison exists
[ ] Stable reduction and non-finite checks exist
[ ] Gradient triangulation passes, if training
[ ] Effective-batch numerator/denominator and accumulation boundary are explicit
[ ] AMP unscale, finite check, clipping, step, and scale-update order is tested
[ ] Optimizer/state trace passes, if stateful
[ ] Checkpoint reproduces the declared next step, not only a load round trip
[ ] Mid-accumulation checkpoint policy is explicit
[ ] Serialization round trip passes, if persistent
[ ] Framework schema and metadata checks pass, if integrated
[ ] Strict and sanitizer builds pass on covered paths
[ ] Complete model outputs pass over the declared dynamic envelope
[ ] Decision boundaries, task metrics, slices, and denominators are replayed
[ ] Stateful paths pass chunk/full and required-horizon comparisons
[ ] Attention masks state polarity, broadcast axes, and absolute-position alignment
[ ] KV cache owner, length/capacity, update timing, reset, and stream identity are tested
[ ] CPU/GPU or backend differential tests pass
[ ] Practice environment chosen for the question, not merely the available accelerator
[ ] Ephemeral files, billing, shutdown, and artifact copy-out plan recorded
[ ] Benchmark scope, shapes, environment, and variance recorded
[ ] Residual limitations and NOT_RUN branches stated
```

# Primary reference shelf

The guide intentionally relies on current primary documentation for version-sensitive behavior. Links and current-behavior statements in this shelf were rechecked on 2026-07-13. Stable documentation URLs may advance to newer releases, so preserve exact package, compiler, runtime, and service records with executable claims:

- [PyTorch `torch.compile` introduction](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [PyTorch common graph breaks](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.common_graph_breaks.html)
- [PyTorch `torch.export`](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html)
- [PyTorch CPU threading](https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html)
- [PyTorch backend diagnostics and oneDNN verbose](https://docs.pytorch.org/docs/stable/backends.html)
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
- [PyTorch testing utilities](https://docs.pytorch.org/docs/stable/testing.html)
- [scikit-learn model evaluation and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [scikit-learn decision-threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [PyTorch automatic mixed-precision examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch scaled dot-product attention tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)
- [Hugging Face cache explanation](https://huggingface.co/docs/transformers/cache_explanation)
- [Hugging Face cache strategies](https://huggingface.co/docs/transformers/kv_cache)
- [PyTorch optimizer gradient clearing](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html)
- [PyTorch saving and loading general checkpoints](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
- [PyTorch Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)
- [PyTorch custom C++ and CUDA operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [LibTorch stable ABI](https://docs.pytorch.org/docs/stable/notes/libtorch_stable_abi.html)
- [PyTorch stable C++ operator API](https://docs.pytorch.org/cppdocs/api/stable/operators.html)
- [PyTorch ONNX export API](https://docs.pytorch.org/docs/stable/onnx.html)
- [PyTorch functional Python custom operators](https://docs.pytorch.org/tutorials/advanced/python_custom_ops_functional.html)
- [PyTorch custom-operator autograd and registrations](https://docs.pytorch.org/tutorials/advanced/python_custom_ops_registrations.html)
- [PyTorch custom-operator guidance](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
- [Extending the PyTorch ONNX exporter operator support](https://docs.pytorch.org/tutorials/beginner/onnx/onnx_registry_tutorial.html)
- [PyTorch `torch.utils.data`](https://docs.pytorch.org/docs/stable/data.html)
- [PyTorch data-loading optimization tutorial](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html)
- [PyTorch pinned-memory and non-blocking copy guide](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
- [PyTorch multiprocessing best practices](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html)
- [PyTorch `Tensor.record_stream`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)
- [PyTorch packed-sequence utilities](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html)
- [DLPack Python array exchange specification](https://dmlc.github.io/dlpack/latest/python_spec.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
- [ROCm HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
- [oneDNN basic concepts](https://uxlfoundation.github.io/oneDNN/dev_guide_basic_concepts.html)
- [oneDNN memory format propagation](https://uxlfoundation.github.io/oneDNN/page_memory_format_propagation_cpp.html)
- [Netlib SGEMM contract](https://www.netlib.org/lapack/explore-html/dd/d09/group__gemm_ga8cad871c590600454d22564eff4fed6b.html)
- [GCC optimization options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)
- [GCC optimization diagnostics](https://gcc.gnu.org/onlinedocs/gcc/Developer-Options.html)
- [NumPy performance and CPU oversubscription](https://numpy.org/doc/stable/user/basics.performant_code.html)
- [ONNX operators and versioned schemas](https://onnx.ai/onnx/operators/)
- [ONNX shape inference](https://onnx.ai/onnx/repo-docs/ShapeInference.html)
- [ONNX Runtime custom operators](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html)
- [ONNX Runtime C API](https://onnxruntime.ai/docs/api/c/struct_ort_api.html)
- [ONNX Runtime I/O binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html)
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)
- [ONNX Runtime execution providers](https://onnxruntime.ai/docs/execution-providers/)
- [ONNX Runtime thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
- [ONNX Runtime performance tuning](https://onnxruntime.ai/docs/performance/tune-performance/)
- [ONNX Runtime quantization and accuracy debugging](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [MLPerf Inference benchmark definitions and quality targets](https://mlcommons.org/benchmarks/inference-datacenter/)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Google Colab local runtimes](https://research.google.com/colaboratory/local-runtimes.html)
- [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks)
- [GitHub Codespaces overview](https://docs.github.com/codespaces/overview)
- [GitHub Codespaces billing](https://docs.github.com/en/billing/concepts/product-billing/github-codespaces)
- [GitHub Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions)
- [Hugging Face Spaces overview](https://huggingface.co/docs/hub/spaces-overview)
- [Hugging Face ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)
- [GPU-accelerated ML in WSL](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)
- [MSVC binary compatibility 2015–2026](https://learn.microsoft.com/en-us/cpp/porting/binary-compat-2015-2017?view=msvc-170)
- [Potential errors passing CRT objects across DLL boundaries](https://learn.microsoft.com/en-us/cpp/c-runtime-library/potential-errors-passing-crt-objects-across-dll-boundaries?view=msvc-170)
- [MSVC DLL export with `__declspec(dllexport)`](https://learn.microsoft.com/en-us/cpp/build/exporting-from-a-dll-using-declspec-dllexport?view=msvc-170)
- [MSVC decorated names](https://learn.microsoft.com/en-us/cpp/build/reference/decorated-names?view=msvc-170)
- [Windows DLL search order](https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order)
- [Latest supported Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)
- [MSVC `_aligned_malloc`](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/aligned-malloc?view=msvc-170)
- [DUMPBIN reference](https://learn.microsoft.com/en-us/cpp/build/reference/dumpbin-reference?view=msvc-170)
- [MSVC AddressSanitizer](https://learn.microsoft.com/en-us/cpp/sanitizers/asan?view=msvc-170)
- [CMake `MSVC_RUNTIME_LIBRARY`](https://cmake.org/cmake/help/latest/prop_tgt/MSVC_RUNTIME_LIBRARY.html)
- [CMake Presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html)
- [CTest](https://cmake.org/cmake/help/latest/manual/ctest.1.html)
- [uv project workflows](https://docs.astral.sh/uv/guides/projects/)
- [Development Container Specification](https://containers.dev/implementors/spec/)
- [PyTorch local installer](https://pytorch.org/get-started/locally/)
- [GitHub Actions setup-python](https://github.com/actions/setup-python)
- [Clang AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html)
- [Jupyter](https://jupyter.org/)
- [pybind11 NumPy and buffer protocol](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html)
- [LiteRT INT8 quantization specification](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/quantization_spec)
- [CMSIS-NN current documentation](https://arm-software.github.io/CMSIS-NN/latest/index.html)
- [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN)
- [LAPACK scaled sum-of-squares (`LASSQ`)](https://www.netlib.org/lapack/explore-html/d8/d76/group__lassq.html)

The references support contracts and tool behavior. The reported experiments support only the specific implementations and environments described.
