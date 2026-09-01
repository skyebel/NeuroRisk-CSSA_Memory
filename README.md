# NeuroRisk: EEG-Based Neurological Risk Screening

**Cognitive Science Student Association (CSSA) Memory Group | 2025-2026**

NeuroRisk is a research and prototyping project investigating whether
low-cost, dry-electrode EEG hardware combined with machine learning can
provide an accessible first-line screening tool for neurodegenerative and
neurological conditions that affect memory, specifically Alzheimer's
disease, Huntington's disease, and epilepsy. The project combines a
literature-grounded problem analysis, a trained classifier evaluated on a
public EEG dataset, and a desktop application prototype demonstrating the
intended patient- and clinician-facing workflow.

This repository is a research and design artifact. It is not a validated
medical device and is not intended for clinical diagnostic use. See
[Disclaimer](#disclaimer) below.

## Team

| Name | Focus Area |
|---|---|
| Tessa Pizzo | Cognitive Science, Neuroscience |
| Audrey Perillo | Cognitive Science, Neuroscience |
| Angus Liu | Cognitive Science, Machine Learning |
| Parisa Emam | Data Science |
| Nitika Bhawe | Cognitive Science, Neuroscience |
| Skye Belsher | Math-CS + Cognitive Science, Machine Learning |

## Motivation

High-end diagnostic imaging for neurodegenerative disease, primarily MRI
and PET, is concentrated in urban medical centers, costly, and often only
partially covered by insurance. These barriers disproportionately affect
rural and low-resource populations:

- Only 776 of 24,658 U.S. neurology providers (3.1 percent) practice in
  rural locations, and only 20 percent of rural counties have at least one
  neurologist (Kurek et al., 2024; McGinley et al., 2024).
- Standard imaging costs range from approximately $1,200-$4,000 for MRI and
  $3,000-$7,000 for PET, with PET scans often not fully covered when used
  for screening (Centers for Medicare & Medicaid Services, 2025).
- Roughly 9.5 percent of households in low-access areas do not own a
  vehicle, compounding travel and time-off-work costs (Owens et al., 2022).
- Cognitive decline is frequently dismissed as normal aging, and stigma
  around testing contributes to substantial underdiagnosis: an estimated
  75 percent of people with epilepsy in low-income countries do not
  receive needed treatment, and up to 75 percent of dementia cases
  worldwide are never formally diagnosed (WHO, 2019; WHO, 2023;
  Livingston et al., 2020).

Portable, dry-electrode EEG systems cost on the order of $5,000-$20,000,
require no gel or scalp preparation, and can be operated by
non-specialist staff in a community clinic or mobile health unit. The
central hypothesis of this project is that a trained classifier applied
to such EEG recordings, paired with a structured symptom questionnaire,
can support earlier and more accessible risk stratification than is
currently available to underserved populations.

Full citations are listed in [References](#references).

## Repository Contents

| File | Description |
|---|---|
| `neuro_screen.py` | Desktop prototype (Tkinter) implementing the patient risk-assessment workflow, live EEG acquisition panel, dry-EEG hardware catalog, longitudinal history tracking, and accessibility and data-privacy modules. |
| `AD_EEG_SVM_v12.ipynb` | Model development notebook: spectral feature extraction and a nested cross-validated SVM classifier for Alzheimer's disease detection from resting-state EEG. |
| `CSSA_Memory_Project_NeuroRisk.pdf` | Project pitch deck covering the background research, target datasets, proposed system, and deployment plan. |

## Target Conditions and Datasets

The project scopes three conditions with distinct EEG signatures and
distinct target populations:

**Alzheimer's disease.** EEG slowing is the established biomarker:
increased power in the delta and theta bands with decreased power in the
alpha and beta bands, consistent with cholinergic deficit and cortical
disconnection. Modeled in this repository using the UCI/OpenNeuro
resting-state EEG dataset (ds004504), comprising EEG recordings for
Alzheimer's and cognitively normal subjects.

**Huntington's disease.** Associated with cortical slowing, reduced alpha
band amplitude, and increased delta activity. The target population
includes individuals who have tested positive for the HTT gene mutation
but remain pre-manifest, for whom frequent low-cost monitoring is
impractical with MRI or PET. Reference datasets identified for future
model development: Enroll-HD (30,511 participants) and the Lancaster
University Huntington's disease and controls EEG dataset.

**Epilepsy.** Characterized by recurrent seizures; temporal lobe damage
from recurrent seizure activity is associated with retrogade memory
impairment. Reference dataset identified for future model development:
the CHB-MIT EEG dataset.

At the current stage, a trained and cross-validated classifier exists only
for Alzheimer's disease (see [Model Performance](#model-performance)
below). The desktop application's screening logic for Huntington's disease
and epilepsy uses a Gaussian likelihood model over literature-informed band
power profiles and is a placeholder for future data-driven models trained
on the datasets above.

## Model Development (`AD_EEG_SVM_v12.ipynb`)

The notebook implements an end-to-end pipeline for Alzheimer's disease
classification from resting-state EEG:

1. **Acquisition.** Downloads the OpenNeuro ds004504 dataset (eyes-closed
   resting EEG, 19-channel 10-20 montage) directly via the OpenNeuro S3
   bucket.
2. **Preprocessing.** Bandpass filter (1-40 Hz) with a 50 Hz notch filter,
   6-second sliding windows with 50 percent overlap, and amplitude-based
   artifact rejection.
3. **Feature extraction.** Welch power spectral density per channel and
   window, aggregated into log-absolute power, relative (normalized)
   power, per-band power across the five canonical EEG bands (delta,
   theta, alpha, beta, gamma), left/right hemispheric asymmetry, and a
   frontal/occipital theta-to-alpha ratio.
4. **Classification.** A support vector machine (RBF and linear kernels)
   with PCA whitening, trained and evaluated with nested 5-fold
   cross-validation: an inner loop performs grid search over C and gamma
   using balanced accuracy as the selection criterion, and an outer loop
   provides an unbiased performance estimate.

### Model Performance

The nested cross-validation procedure reports a mean balanced accuracy of
approximately 80 percent across folds. Balanced accuracy is used
throughout to account for class imbalance between the Alzheimer's and
cognitively normal groups. Per-fold results, a confusion matrix, and
per-subject classification outcomes are generated in the results
dashboard cell of the notebook.

This result should be interpreted as a research-stage benchmark on a
single public dataset, not a validated diagnostic sensitivity or
specificity. See [Disclaimer](#disclaimer).

## Application Prototype (`neuro_screen.py`)

A Tkinter desktop application demonstrating the intended end-to-end
product experience:

- **Risk Assessment.** Combines an uploaded or live EEG recording with a
  structured health questionnaire to produce a per-condition risk score.
- **Live EEG integration.** Connects to a real headset through BrainFlow
  (OpenBCI, Muse, Cyton, Ganglion, and other supported boards) or a raw
  serial OpenBCI protocol, with a microphone-based fallback that is
  explicitly labeled as a non-EEG demonstration mode when neither is
  available. Band powers are computed with per-band IIR filters and
  streamed to the interface in real time from a background thread.
- **EEG hardware catalog.** A filterable directory of affordable
  dry-electrode EEG headsets with specifications, compatible SDKs, and
  purchase links, intended to help clinics and individual users select
  compatible hardware.
- **Longitudinal tracking.** A patient history dashboard that stores
  assessment results locally and visualizes score trends over time and
  across conditions, including support for multiple patient profiles
  under the same installation, relevant to tracking hereditary risk in
  Huntington's disease.
- **Accessibility.** Configurable text scale, high-contrast theme, a
  dyslexia-friendly font option, and language settings.
- **Data privacy.** A dedicated panel documenting the application's data
  retention policy and its alignment with HIPAA, a GDPR-aligned privacy
  policy, and ISO 27001 as a security reference standard, along with
  user-facing controls for data access, erasure, and export.

### Requirements

```
python >= 3.9
tkinter          (standard library on most platforms)
numpy, scipy      (optional, enables live EEG signal processing)
brainflow         (optional, enables real EEG headset support)
pyserial          (optional, enables raw serial OpenBCI support)
sounddevice       (optional, enables microphone demo mode)
```

The application degrades gracefully when optional dependencies are
absent: live EEG features are disabled and the affected panels indicate
that the relevant package should be installed.

### Running the Application

```bash
pip install numpy scipy brainflow pyserial sounddevice
python neuro_screen.py
```

## Deployment Plan

The project proposal outlines a phased rollout:

1. **UCSD pilot.** Test the prototype with students, researchers, labs,
   and faculty to refine accuracy, usability, and EEG data workflows.
2. **Local deployment.** Partner with local clinics and community health
   organizations, including the Presbyterian Foundation, to trial the
   tool in a clinical setting.
3. **Rural U.S. expansion.** Support telehealth delivery and partner with
   mobile health clinics serving communities with limited neurology
   access.
4. **Long-term.** Collaborate with international health organizations
   such as WHO and USAID to expand multilingual, low-cost screening
   globally.

## Disclaimer

NeuroRisk is a student research and design project. The application and
the associated classifier are prototypes intended to demonstrate a
proposed screening workflow and to evaluate feasibility on public
datasets. They have not been clinically validated, are not FDA-cleared or
CE-marked, and must not be used to diagnose, rule out, or guide treatment
of any medical condition. Any real-world deployment would require formal
clinical validation, regulatory review, and IRB-approved prospective
studies prior to use in patient care.

## References

- Alzheimer's Disease International. (2021). *World Alzheimer Report*.
- Centers for Medicare & Medicaid Services (CMS). (2025). *Medicare
  coverage of diagnostic imaging services*.
- Frank, R. G. (2014). *Behavioral economics and health policy*.
- GoodRx. (2024). *Cost of MRI and PET imaging in the United States*.
- Kurek, K., et al. (2024). *Geospatial access to neurologists in the
  United States*.
- Livingston, G., et al. (2020). Dementia prevention, intervention, and
  care: 2020 report of the Lancet Commission. *The Lancet*.
- Martinez, J., et al. (2020). *Temporal lobe epilepsy and memory
  outcomes*.
- McGinley, M. P., et al. (2024). *Neurology workforce shortages in
  nonmetropolitan counties*.
- Owens, P. L., et al. (2022). *Rural-urban disparities in access to
  neurological care*.
- Rafferty, A. P., et al. (2025). *Geographic access to specialty care*.
- Viskontas, I. V., et al. (2000). *Memory impairment in temporal lobe
  epilepsy*.
- World Health Organization (WHO). (2019). *Epilepsy: a public health
  imperative*.
- World Health Organization (WHO). (2023). *Dementia fact sheet*.
- World Health Organization (WHO). (2024). *Epilepsy fact sheet*.

Dataset citations:

- OpenNeuro ds004504: resting-state EEG in Alzheimer's disease, frontotemporal
  dementia, and healthy control subjects.
- CHB-MIT Scalp EEG Database (PhysioNet).
- Enroll-HD: a global clinical research platform for Huntington's disease.
- Lancaster University Huntington's Disease and Controls EEG Dataset.
