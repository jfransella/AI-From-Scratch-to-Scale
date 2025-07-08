# Hopfield Network (Model 04) – Empirical Analysis Plan

## 2.1 Backward Link

This document details the practical implementation and performance analysis of the model. For the full historical context and theoretical background, please refer to our **[Theoretical Deep Dive](./01_deep_dive.md)**.

## 2.2 Proposed "Success Case" Experiment

**Objective:** To demonstrate the Hopfield network’s strength in content-addressable memory, we will design a classic pattern-recall experiment. The goal is to show that the network can successfully store multiple patterns and retrieve a complete pattern from a partial or corrupted input cue.

**Dataset/Patterns:** Rather than a conventional labeled dataset, we will use a set of *binary patterns* specifically crafted for auto-associative memory tests. For example, we can create a few simple 2D binary images – say $5 \times 5$ pixel grids representing capital letters (like a blocky **“H”**, **“E”**, and **“L”**). Each letter pattern will be a binary vector of length 25 (with +1 for black pixels and -1 for white). These distinct patterns will be our “memories” to store in the Hopfield network. We choose such patterns because they are easy to visualize and distinctly different, yet small enough for the Hopfield net to handle. (Alternatively, one could use simpler abstract patterns or bit-vectors of length N=100 with random assignments. The key is to ensure the patterns are mutually uncorrelated enough so the network can store them simultaneously.)

**Method:** Using the Hebbian learning rule, we will encode all chosen patterns into the weight matrix of a Hopfield network. Once the network is “loaded” with these memories, we will conduct recall tests. For each stored pattern, we will take a *noisy version* as input: for instance, flip 10–20% of the bits (turn some 1s to -1s or vice versa) to simulate corruption, or erase parts of the letter (e.g. remove a stroke of the “H”). We will initialize the network’s neurons to this partial/noisy pattern and then let it run its asynchronous update dynamics.

**Expected Outcome:** If the Hopfield network is functioning as intended, it should **converge to the correct stored pattern** despite the noise. For example, if we present a corrupted “H” pattern, after a series of updates the network’s state should end up *exactly matching the pristine “H”* that was originally stored. This demonstrates the network’s ability to do error correction and pattern completion. The retrieved pattern is the network’s output.

**Metrics:** We will measure:

* *Convergence Success Rate*: the percentage of trials in which the network converges to a correct stored pattern (as opposed to a wrong pattern or an ambiguous state) for a given level of noise. We expect a high success rate when the noise level is low-to-moderate, illustrating robustness.
* *Convergence Time (Iterations)*: how many update steps it takes on average for the network to settle on a pattern. Hopfield networks typically converge quickly (in a few iterations for small N). We can track this to verify the theoretical fast convergence property.
* *Hamming Distance Reduction*: For each test, measure the Hamming distance (number of differing bits) between the network’s state and the nearest stored pattern, initially and after convergence. We expect to see the distance drop to 0 for successful recalls. A plot of Hamming distance over iterations would vividly show the network “cleaning up” the pattern.
* *Energy Landscape Visualization*: Although more abstract, we can compute the network’s energy \$E\$ (as defined in the deep dive) at each iteration. We expect to see \$E\$ monotonically decreasing and finally reaching a minimum. Plotting energy vs. time would confirm that the network is indeed descending into an energy valley corresponding to a stored memory.

**Visualizations:** We will include images to illustrate the recall process:

* An example set of the **stored patterns** (e.g., the letter images).
* A **noisy input** presented to the network.
* The **network’s reconstructed output** after convergence.

For instance, a figure might show a 5x5 letter “H” with several pixels flipped (as input) and the fully restored “H” after the Hopfield network’s recall. Another visualization could be a trajectory of the network’s state: starting from the noisy pattern, after 1 iteration some pixels correct, after 2 iterations more correct, etc., until it matches the memory. Such step-by-step images would confirm qualitatively that partial information is enough for the Hopfield model to retrieve complete information – the hallmark of its success.

By using simple binary patterns and visualizing the outcomes, this experiment will clearly demonstrate the **success case** for Hopfield Networks: robust associative memory retrieval. We anticipate results showing nearly 100% accuracy in recalling the original patterns when the distortion is within the capacity, and the network settling into the right memories within only a handful of updates.

## 2.3 Proposed "Failure Case" Experiment

No model is without limitations. For Hopfield networks, the most famous limitation is their **restricted storage capacity and the emergence of spurious states** when that capacity is exceeded. Our failure-case experiment will explicitly expose this weakness.

**Scenario:** We will push the Hopfield network beyond its comfortable memory capacity by trying to store *either too many patterns or highly interfering patterns*, and observe the network’s breakdown in performance. There are two complementary approaches we can take:

1. **Capacity Overload Test:** Gradually increase the number of stored patterns $P$ for a fixed-size network and measure recall performance.
2. **Similar Pattern Interference Test:** Store patterns that are very similar to each other (e.g. one pattern and a slightly modified version of it) and see if the network confuses them.

For definiteness, we’ll start with the capacity overload approach.

**Dataset:** We will generate a series of random binary patterns of length N (say $N = 100$ neurons for a moderately sized network). These could be simply random 100-bit vectors with equal probability of ±1 for each bit. Random patterns are actually the hardest to store in Hopfield nets when there are many, because they are uncorrelated and utilize maximum capacity. We will start by storing a small number of them, then increase the number $P$. According to theory, the critical capacity is about $0.14N$ patterns for reliable recall – for $N=100$, that’s about 14 patterns. We will test the network with $P$ around that range and beyond (e.g. $P = 5, 10, 15, 20, 30$ patterns).

**Method:** For each number of stored patterns $P$, we construct the weight matrix using the Hebbian rule with those $P$ patterns. Then we will attempt to recall each of the $P$ patterns by presenting either the exact pattern or a slightly noisy version of it as initial state and see what the network converges to. We’ll track whether it converges to the correct memory, an incorrect memory, or a spurious state (i.e., a stable state that was never one of the training patterns). We will repeat multiple trials (with different random sets of patterns) to get statistical reliability.

**Expected Outcome:** Initially, when $P$ is low (well below capacity), the network should recall each pattern perfectly, even from a noisy start – as seen in the success case. However, as $P$ grows close to or beyond the capacity (\~14 for N=100), we expect to see **failures**:

* The network might converge to **wrong memories**: e.g., we present pattern #3 but the network ends up in pattern #7, because the input was closer to #7 in some bits or because #3 and #7 interfere.
* Worse, the network might settle into **spurious patterns** that were never inserted as memories. These could be hybrid mixes of two or more real patterns (for example, half the bits from one memory and half from another), or a seemingly random pattern that is a local minimum of the energy but not one of the taught patterns.
* The network may also fail to converge within a reasonable time, oscillating or cycling among states (especially if we used synchronous updates or if symmetry is broken), but with asynchronous updates a cycle is rare unless weights are asymmetric. Typically, failure manifests as convergence to an incorrect stable state.

We expect that when $P > 0.14N$, the **recall accuracy** will drop significantly. We will measure this accuracy: the fraction of test attempts that end up in the correct pattern. For example, with $P = 5$ (well below capacity), accuracy might be \~100%. By $P = 15$ (at capacity), it might still be high if those patterns happened to be not too interfering. But by $P = 25$ or 30 (double capacity), accuracy could plummet, perhaps to 50% or worse – essentially the network starts “forgetting” or confusing patterns. We will also measure the occurrences of spurious states. One way is: after convergence, check if the final state matches any of the stored patterns; if not, that’s a spurious result. We might find, for instance, that by $P = 30$, *most* recalls end up in states that aren’t in the memory list (indicating the net is mostly producing false memories).

**Diagnostic Metrics:**

* Plot **Recall Accuracy vs. Number of Stored Patterns**. This should show a clear degradation beyond the theoretical capacity \~0.14N. We expect a curve that starts near 100% and dips downward as P increases, possibly dropping sharply after \~15 patterns for N=100.
* Count of **Spurious Recall** events. We might report what percentage of trials resulted in a state that wasn’t one of the original patterns. This underscores the confusion the network experiences when overloaded.
* If feasible, visualize examples of spurious patterns (especially if our patterns have some structure like images). For instance, if we stored several letter patterns, a spurious retrieval might look like a nonsensical mix (maybe a pattern that is like an “H” merged with an “E”). Visualizing such an outcome would qualitatively show the failure mode – the network produces a “memory” that was never learned, an emergent false memory.
* Additionally, measure **energy of correct vs spurious states**. Often spurious states are local minima with slightly higher (less optimal) energy than true memories. We could see if the network sometimes gets stuck in a shallow local minimum rather than the deeper one corresponding to a true pattern.

For the **similar pattern interference test** (as an extra exploration), we could store intentionally overlapping patterns (like pattern X and pattern X with a small modification). The expectation is the network might confuse them – e.g., starting from a cue that’s ambiguous, it might randomly fall into one or the other, or settle on an intermediate mix. This highlights that Hopfield nets have trouble discriminating patterns that aren’t well separated in the input space.

**Outcome:** The failure case will be demonstrated when we see the Hopfield network **mis-recalling memories**. For example, we might present the network with a noisy version of pattern 1, but it ends up outputting pattern 2 – clearly a failure for an associative memory. Or, we start with a cue for pattern 1 and the network settles into some pattern 1/2 hybrid that we can visibly tell was not one of the originals. These outcomes align with known limitations: as memories pile up, the crosstalk between stored patterns via shared weights causes the wrong attractors to form.

In summary, this experiment will likely show that beyond a certain point, **adding more memories actually degrades performance dramatically** – a phenomenon unique to associative memory networks. The expected result is a clear illustration that the Hopfield network cannot scale arbitrarily; it has a finite capacity and when pushed past it, it **fails** by retrieving incorrect or nonexistent patterns (an analog to human memory confusion or false memories). Such empirical evidence of failure will set the stage to discuss why we need more advanced models to overcome these limitations.

## 2.4 The Transition Narrative

The weakness exposed in the failure case is *exactly* the problem that **LeNet-5** was designed to solve. It addresses this by introducing a multi-layered, hierarchical learning architecture (the convolutional neural network) that can scale to complex patterns like handwritten digit images. In contrast to the Hopfield net’s brittle memory capacity and inability to generalize, LeNet-5 uses **feature extraction (convolution/pooling layers)** and **supervised learning (backpropagation)** to handle large-scale image recognition reliably. Hopfield networks could only recall what they literally stored and would fail on new or excessive inputs, but LeNet-5 was built to *learn from examples* and generalize to new data – effectively overcoming the associative memory’s limits by moving to a deeper, trainable model with far greater capacity for variation. LeNet-5’s convolutional design specifically tackles the kind of task a Hopfield net cannot do: recognizing many different patterns (like 10 handwritten digit classes) in a robust way, even with shifts or distortions, something that requires generalization beyond memorization. Thus, the shortcomings of the Hopfield network directly motivated the development of more powerful architectures like LeNet-5, which introduces **layered feature learning and translation-invariance** to solve pattern recognition at scale.
