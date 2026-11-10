---
title: "Portfolio"
date: "2024-01-01T00:00:00Z"
layout: single
---

Selected projects across data platforms, ML systems, and data products.

---

## AI-Native Data Function (Stream)

**A data team and platform where people and AI agents work from the same trusted definitions.**

As Head of Data at Stream (formerly Wagestream), a financial wellness fintech, I own the data function: data platform, analytics engineering, data science, BI, governance, and data products. I report to the CPTO.

**Key achievements**:
- Designed and built a hub-and-spoke data team: a lean core plus analytics engineers embedded in Operations, Credit and the US, so that key parts of the business have a dedicated data partner
- Designed a Junior-to-Staff levelling framework across the Analytics Engineering and Data Science tracks
- Built an MCP-based semantic context layer that gives AI agents governed access to metric definitions, schema documentation and lineage, so self-serve analytics and agentic coding run on trusted definitions
- Launched an in-house, AI-first BI platform that replaced the incumbent commercial tool, as part of a data-stack cost-optimisation programme on track to save ~$150k/year
- Created real-time UK macroeconomic indicators from Stream's proprietary data and shared them with senior external stakeholders

**Stack**: Snowflake, dbt, Python, Streamlit, MCP, AWS

---

## Open Banking Data Platform (Tarabut)

**Enterprise data platform powering data products across 4 jurisdictions.**

As the first data hire at Tarabut, I designed and delivered the company's Enterprise Data Platform from scratch. The platform unified data from UK and Middle East operations across AWS and GCP, enabling both internal analytics and external data products.

**Key achievements**:
- Scaled data function from 1 to 10 FTEs across 4 countries
- Delivered ML-powered bank transaction enrichment services deployed across 6 regions
- Turned platform data into external data products that opened new revenue streams
- Led the post-M&A integration that put UK and Middle East data on one platform
- Chaired the Data Governance Committee and set one governance framework for the whole group

**Stack**: AWS, GCP, Snowflake, dbt, Python, Kubernetes, Airflow

---

## Financial Behaviour ML Models (TrueLayer)

**Production ML services powering the UK's first open banking unicorn.**

As the joint-first data hire at TrueLayer (~employee #20), I built the company's first ML-powered services from the ground up. These models extracted financial insights from raw bank transactions to power lending, affordability, and verification use cases.

**What I built**:
- **Transaction categorisation**: Automated classification of bank transactions enabling spend analysis
- **Income estimation API**: ML model inferring income from transaction patterns - shipped as a commercial product
- **Affordability signals**: Rent detection, recurring payment identification, and spending pattern analysis

**Impact**: 99.9% availability with weekly model updates at scale. These became TrueLayer's first revenue-generating data products beyond the core API.

**Stack**: Python, Scikit-learn, TensorFlow, Kubernetes, AWS

---

## AI Doom Boom Bias Tracker

**A live NLP application tracking media bias in AI news coverage.**

I built an end-to-end data platform on GCP to analyse how news outlets frame AI - whether as transformative ("boomer") or existential risk ("doomer"). The system ingests articles daily via NewsAPI, classifies them using a zero-shot transformer model (facebook/bart-large-mnli), and surfaces trends through an interactive dashboard.

**Architecture**: Cloud Functions → BigQuery → Cloud Run (Streamlit)

**Stack**: Python, BigQuery, Cloud Functions, Cloud Scheduler, Hugging Face Transformers, Streamlit, Docker

[Live App](https://streamlit-app-824805393106.europe-west2.run.app/) ・ [GitHub](https://github.com/alejio/news-bias-detection) ・ [Technical Write-up](/insights/2024-10-20-ai-doom-boom-bias-tracker/)

---

## Consumer Trend Prediction (Black Swan Data)

**ML pipeline processing 100M+ social media posts monthly for Fortune 500 clients.**

Led Data Science development for a B2B SaaS platform that predicted consumer trends by analysing social media signals. The system identified emerging trends weeks before they appeared in traditional market research.

**Clients**: Fortune 500 CPG and retail companies

**Stack**: Python, NLP, distributed computing

---

## Seismic Data ML Automation (PGS)

**Applied ML to geophysical data processing, reducing QC time by 70%.**

During my time as a geophysicist, I introduced machine learning techniques to automate quality control processes in seismic data swell noise attenuation. This work was novel enough to be published and presented at a major industry conference.

**Publication**: "Using Statistical Techniques to Improve the QC Process of Swell Noise Filtering" - EAGE London 2013

**Impact**: 70% reduction in manual QC time for TB-scale seismic datasets

**Stack**: R, MATLAB, signal processing
