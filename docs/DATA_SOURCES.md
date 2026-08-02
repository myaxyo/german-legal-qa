# Data Sources

## Overview

All data sources used in this project are publicly available under permissive licenses suitable for academic research.

## 1. Federal Statutes (Gesetze im Internet)

- **URL**: https://www.gesetze-im-internet.de
- **Format**: XML (per-law ZIP archives)
- **Coverage**: All current German federal laws (~6,000 statutes)
- **License**: Public domain (official government publications)
- **Key laws**: BGB, StGB, GG, HGB, ZPO, StPO, ArbGG, AGG, MuSchG

### Structure
Each statute is an XML file containing:
- `<norm>` elements for each section (§)
- `<enbez>` for the section number
- `<titel>` for the section title
- `<textdaten>` for the legal text

## 2. Court Decisions (Open Legal Data)

- **URL**: https://de.openlegaldata.io
- **Format**: JSON API (paginated)
- **Coverage**: 100,000+ anonymized German court decisions
- **License**: CC-BY 4.0
- **Courts**: BGH, BVerfG, various OLGs and LGs

### Fields used
- `court.name`: Issuing court
- `date`: Decision date
- `file_number`: Case reference number (Aktenzeichen)
- `content`: Full decision text (HTML)

## 3. GerLayQA (Layperson Legal QA)

- **URL**: https://huggingface.co/datasets/fhswf/GerLayQA
- **Format**: HuggingFace Dataset
- **Coverage**: German legal questions by citizens with expert answers
- **License**: Research use
- **Categories**: Civil law, criminal law, labor law, family law, etc.

### Fields
- `question`: Citizen's legal question (plain German)
- `answer`: Lawyer's answer
- `category`: Legal domain
- `split`: train/test

## 4. Potential Additional Sources

| Source | Description | Status |
|--------|-------------|--------|
| EUR-Lex | EU legislation (German translations) | Planned |
| dejure.org | Law cross-references | To evaluate |
| C-DBR | Compiled German federal law dataset | To evaluate |

## Licensing Summary

| Source | License | Commercial use | Modification |
|--------|---------|----------------|-------------|
| Gesetze im Internet | Public domain | Yes | Yes |
| Open Legal Data | CC-BY 4.0 | Yes | Yes (with attribution) |
| GerLayQA | Research | Academic only | With attribution |

## Data Volume Estimates

| Source | Raw size | Processed chunks | Tokens (approx) |
|--------|----------|-----------------|-----------------|
| Statutes | ~2GB XML | ~200K sections | ~50M tokens |
| Decisions | ~5GB JSON | ~500K chunks | ~200M tokens |
| GerLayQA | ~50MB | ~10K QA pairs | ~5M tokens |
| Synthetic QA | Generated | ~50K pairs | ~25M tokens |
