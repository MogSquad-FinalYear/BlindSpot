# Detector Router — Technical Working

Mathematical specification of the rule-based detector selection implemented in
`rule_engine.py`, with thresholds from `config.py` and calibration data from
`paper/scripts/highpass_energy.py`.

Revision basis: 2026-08-29 (native resolution dropped from the routing decision).

---

## 1. The one feature: high-frequency spectral log-energy

Given an uploaded RGB image $I$ of size $W \times H$:

### Step 1 — Luma conversion and fixed-size resample

PIL's `"L"` conversion is the ITU-R 601-2 luma, followed by a bicubic resize to a
fixed $N = 256$ grid (Pillow's `resize` default resample):

$$g[y,x] \;=\; \frac{1}{255}\,\mathrm{Resize}_{N\times N}\!\big(0.299R + 0.587G + 0.114B\big),
\qquad g \in [0,1]^{256\times 256}$$

The resize is what makes the feature resolution-independent — the frequency axis is
always normalised to the 256-grid, never to the upload's native pixel count.

### Step 2 — Centred 2-D DFT

$$F[u,v] \;=\; \mathrm{fftshift}\left(\sum_{y=0}^{N-1}\sum_{x=0}^{N-1}
g[y,x]\, e^{-2\pi i (uy+vx)/N}\right)$$

After the shift, DC sits at $(u_0, v_0) = (N/2, N/2) = (128, 128)$.

### Step 3 — Log-magnitude

$$M[u,v] \;=\; \ln\big(|F[u,v]| + 10^{-8}\big)$$

### Step 4 — Outer-radius mask

With $r[u,v] = \sqrt{(u-128)^2 + (v-128)^2}$, $r_{\max} = \tfrac{\sqrt{2}\,N}{2}$ and
`TOP_FRAC` $= 0.2$:

$$\Omega \;=\; \Big\{(u,v) \;:\; r[u,v] \;\ge\; (1-0.2)\cdot \tfrac{\sqrt{2}\cdot 256}{2}
\;=\; 144.815 \Big\}$$

### Step 5 — The scalar

$$E_{\mathrm{hf}} \;=\; \frac{1}{|\Omega|}\sum_{(u,v)\in\Omega} M[u,v],
\qquad |\Omega| = 5797 \text{ bins } (8.85\% \text{ of the plane})$$

### Two properties of the mask worth stating explicitly

**The band is diagonal-only.** The cut radius 144.8 px *exceeds* the axis Nyquist
radius of $N/2 = 128$. So $\Omega$ is not an annulus — it is four corner caps,
covering normalised radial frequency

$$1.131\,f_{\text{Nyq}} \;\le\; f \;\le\; 1.414\,f_{\text{Nyq}}$$

i.e. 0.566–0.707 cycles/pixel measured diagonally. The router reads the *corners* of
the spectrum, exactly where GAN transposed-convolution upsampling leaves its largest
deficit.

**It is log-linear in gain, not contrast-invariant.** Since
$|\mathcal{F}\{a \cdot g\}| = a\,|\mathcal{F}\{g\}|$, an unclipped exposure gain $a$
shifts the feature additively:

$$E_{\mathrm{hf}} \;\mapsto\; E_{\mathrm{hf}} + \ln a$$

Verified on `ai.png`: $E_{\mathrm{hf}} = 1.9555$ at $a = 1$, and $1.2553$ at $a = 0.5$
(predicted $1.9555 - \ln 2 = 1.2624$; the 0.007 gap is uint8 requantisation). At
$a = 2$ it *falls* to 1.5779 because highlights clip. Heavy under-exposure is
therefore the one transform that can push a real photo toward the GAN branch.

---

## 2. The routing function

Let $s = \min(W, H)$ (native, pre-resize). With $\theta_{\text{NPR}} = -1.0$,
$\theta_{\text{UFD}} = -0.3$ and $s_{\max} = 300$ from `config.py`:

$$D(E_{\mathrm{hf}}, s) \;=\;
\begin{cases}
\texttt{npr} & E_{\mathrm{hf}} < -1.0 \\[6pt]
\texttt{universalfakedetect} & -1.0 \le E_{\mathrm{hf}} < -0.3 \;\wedge\; s < 300 \\[6pt]
\texttt{genimagepp\_cliplora} & \text{otherwise}
\end{cases}$$

Note the asymmetry: $s$ appears **only** in the middle branch. It can never route
*away* from the CLIP-LoRA default, only into the narrow UniversalFakeDetect niche —
which is the point of the 2026-08-29 revision.

Downstream, `app.py` flags on a fixed threshold

$$\hat{y} \;=\; \sigma(z) \;>\; 0.2, \qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$

where $z$ is the selected detector's single logit (all three wrappers end in
`torch.sigmoid`).

---

## 3. Threshold calibration

From the 300-real / 300-fake per-generator measurements in
`paper/*/highpass_energy/highpass_energy.csv` (all four copies identical):

| class | $\mu$ | $\sigma$ | $z(\theta = -1.0)$ | $\Pr[E < -1.0]$ |
|---|---|---|---|---|
| BigGAN fake | −1.6204 | 0.2053 | +3.02 | 99.87 % |
| GLIDE fake | −1.2270 | 0.1687 | +1.35 | 91.08 % |
| real (BigGAN set) | 0.8829 | 0.6142 | −3.07 | 0.11 % |
| real (COCO set) | 0.7252 | 0.7678 | −2.25 | 1.23 % |
| flux fake | 0.4963 | 0.5111 | −2.93 | 0.17 % |
| sd3 fake | 0.8344 | 0.5212 | −3.52 | 0.02 % |
| sdxl fake | 0.3450 | 0.6079 | −2.21 | 1.35 % |

Under a Gaussian fit, $\theta_{\text{NPR}} = -1.0$ sits ≈3σ above the BigGAN fake mean
and ≈2.2–3.1σ below every real / DiT mean: recall ≈99.9 % on BigGAN and ≈91 % on
GLIDE, at ≈0.1–1.4 % leakage from real photos and DiT generators.

The separation being exploited is the paired shift $\Delta = \mu_{\text{fake}} - \mu_{\text{real}}$:

| generator | $\Delta$ | % change vs real |
|---|---|---|
| BigGAN | −2.5033 | −283.54 % |
| GLIDE | −2.1403 | −234.35 % |
| flux | −0.2288 | −31.56 % |
| sd3 | +0.1093 | +15.07 % |
| sdxl | −0.3802 | −52.43 % |

The DiT-era shifts are an order of magnitude smaller and sign-inconsistent, which is
exactly why DiT images are *not* routable by this feature and fall to the CLIP-LoRA
default.

The second threshold $\theta_{\text{UFD}} = -0.3$ sits ≈1.9–2.2σ below the real means
(2.6–2.7 % leakage) and above 100 % of both GAN fake distributions, so the band
$[-1.0, -0.3)$ is a genuine low-density gap in every measured distribution — the
"we don't know" region, gated further by $s < 300$.

---

## 4. Live sanity check

Running the actual router over sample uploads:

| image | $W \times H$ | $E_{\mathrm{hf}}$ | branch |
|---|---|---|---|
| `000_biggan_00020.png` | 128×128 | −1.5488 | `npr` |
| screenshot (resized) | 264×340 | 0.3969 | `genimagepp_cliplora` |
| `ai.png` | 676×589 | 1.9555 | `genimagepp_cliplora` |

The BigGAN sample lands at −1.55, within 0.07 of the corpus mean −1.6204, clearing
$\theta_{\text{NPR}}$ by 0.55 ≈ 2.7σ of the fake distribution.

---

## 5. Caveat on the reported probabilities

The per-class σ values are corpus-level, and the $\Pr[E < \theta]$ column assumes
normality. The real-photo distributions ($\sigma \approx 0.61$–$0.77$, over 3× the
fake σ) are almost certainly not Gaussian, being a mixture over textures. Treat those
percentages as an order-of-magnitude leakage estimate, not a calibrated false-positive
rate.

A defensible false-route rate would come from an empirical count of
$\#\{E_{\mathrm{hf}} < -1.0\}$ over the raw per-image energies. `highpass_energy.py`
currently persists only the mean and std per class, so this requires a re-run with
per-image dumping.
