# Vigade analüüs — TÜ kursuste nõustaja (app5.2.py)

## Kokkuvõte

| Mõõdik | Väärtus |
|---|---|
| Katsetuste koguarv | **17** |
| Edukad (👍) | **14** (82,4%) |
| Ebaedukad (👎) | **3** (17,6%) |

| Vahesamm | Mõjutatud halvad juhtumid | % halbadest (n=3) |
|---|---|---|
| **Metaandmete filtreerimine** | 0 | 0% — töötas kõigil juhtudel õigesti |
| **RAG vektorotsing** | 0 | 0% — tagastas korrektelt filtrijärgsed tulemused |
| **LLM vastuse genereerimine** | 3 | **100%** — hallutsineeras kõigil juhtudel |

---

## Halvad juhtumid

| Juhtum | Päring | Metaandmete filtreerimine | RAG vektorotsing | LLM genereerimine |
|---|---|---|---|---|
| **15** | Kunstiga seotud kursused | ✅ Filtreeris õigesti | ✅ Tagastas filtrijärgsed parimad | ❌ Soovitas 3 kursust, mis polnud RAG kontekstis — pidi ütlema et filtritele vastavaid kunstikursusi ei leidu |
| **16** | Arvutimängude mängimine | ✅ Filtreeris õigesti (filtritele vastavaid mängukursusi polnud) | ✅ Tagastas filtrijärgsed parimad | ❌ Soovitas kursuse, mis polnud RAG kontekstis — pidi ütlema et filtritele vastavaid kursusi ei leidu |
| **17** | Jooga | ✅ Jooga kursus filtreeriti õigesti välja (eeldusained olemas) | ✅ Ei tagastanud filtreeritud kursust | ❌ Soovitas filtreeritud kursust (KKSP.03.004) välisest teadmisest — pidi ütlema et filtritele vastavaid kursusi ei leidu |

---

## Kõik katsetused

| # | Päring | Filtrid | Hinnang | Veatüüp |
|---|---|---|---|---|
| 1 | Tahaks leida kursust, kus saab maalida | — | 👍 | — |
| 2 | Tahaksin sissejuhatavat kursust neurovõrkude teemal | — | 👍 | — |
| 3 | Tahaksin õppida arvutite riistvara kohta | — | 👍 | — |
| 4 | Tahaksin lennata droonidega | — | 👍 | — |
| 5 | Tahaksin lennukiga lennata | — | 👍 | — |
| 6 | Tahaksin õppida erinevate protsessorite disaini kohta | — | 👍 | — |
| 7 | Kas saan õppida arvutimänge tegema? | — | 👍 | — |
| 8 | Tahaksin sissejuhatavat kursust neurovõrkude teemal | — | 👍 | — |
| 9 | Soovita mulle kursust, kus saab sporti teha | — | 👍 | — |
| 10 | Mis kursused aitavad mul paremaks juhiks saada? | — | 👍 | — |
| 11 | Kas TÜ-s on kursus "Tehisintellekt ja õigus"? | — | 👍 | — |
| 12 | Soovita mulle kursust teemal "kvantarvutus" | — | 👍 | — |
| 13 | Tahan õppida, kuidas teha ilusat graafilist disaini | — | 👍 | — |
| 14 | Can you recommend me a course about history? | — | 👍* | — |
| 15 | Soovita mulle kunstiga seotud kursusi | kevad + inglise + veebiõpe | 👎 | LLM hallutsineeris |
| 16 | Soovita mulle kursusi, kus saan mängida arvutimänge | kevad + eesti + põimõpe | 👎 | LLM hallutsineeris |
| 17 | Tahaksin teha joogat | sügis + eesti + lähiõpe | 👎 | LLM hallutsineeris |

*\* Päring 14 sai ingliskeelse vastuse — reegli nr 2 rikkumine, kasutaja hindas heaks.*

---
