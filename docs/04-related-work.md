# RELATED WORK — ตัวอย่างงานที่คนอื่นทำ

> รวบรวมเมื่อ: 2026-11-18 · สถานะ: ฉบับแรก (ยังไม่ได้อ่านเปเปอร์ครบทุกใบ)
>
> **จุดประสงค์:** เราเขียน `01`–`03` ขึ้นมาจากหลักการล้วน ๆ โดยไม่ได้ดูว่าคนอื่นทำกันยังไง
> เอกสารนี้คือการตรวจงานตัวเองย้อนหลัง — ข้อไหนที่เราคิดตรงกับงานวิจัย ข้อไหนที่เราพลาด
> และข้อไหนที่เรา *จงใจ* ต่างแล้วต้องอธิบายให้ได้
>
> ทุกข้อมีช่อง **เอาไปใช้** ว่าจะกระทบ phase ไหนของเรา
> เครื่องหมาย ⚠️ = ยังไม่ได้ยืนยันจากตัวเปเปอร์ ได้จาก snippet/abstract เท่านั้น

---

## 1. สรุปสั้นที่สุด

| คำถาม | คำตอบที่ได้จากการสำรวจ |
|---|---|
| มีคนทำ resume parsing + matching เยอะไหม | เยอะมาก ทั้งงานวิจัยและ portfolio บน GitHub |
| มี benchmark สาธารณะให้เทียบไหม | **extraction มี** (ResumeBench, SynthResume) · **matching/ranking แทบไม่มี** — เปเปอร์ ConFit บอกตรง ๆ ว่าไม่มีชุด person-job fit มาตรฐานที่เปิดสาธารณะ |
| สิ่งที่เราทำแล้วคนอื่นไม่ค่อยทำ | ชี้หลักฐานกลับไปยังข้อความต้นฉบับ 100% + วัดเพดานมนุษย์ + ประกาศเป้าก่อนเขียนโค้ด |
| สิ่งที่เราพลาด | ดูข้อ 6 — มี 4 เรื่อง + เทคนิคย่อย 5 อย่างในข้อ 6.5 |

> **🔴 ข้อค้นพบที่ยืนยันแล้วจาก 9 repo (ข้อ 5):** ตรวจ README ของ portfolio project ที่ตรงแนวเรา
> ครบ 9 อัน — **ไม่มีอันไหนเลยที่รายงาน accuracy / precision / recall / ranking quality**
> ทุกอันมี test ที่ตรวจว่า "โค้ดทำงาน" ไม่มีอันไหนมี test ที่ตรวจว่า "ผลถูก"
> อันที่ใกล้สุด (ข้อ 5.3.3) มี resume สมมติ 6 ใบที่ "ควรออกมาเรียงลำดับแบบนี้" ซึ่งเป็น smoke test
> ไม่ใช่ชุดวัดผล · นี่ไม่ใช่ความบังเอิญ มันคือส่วนที่ยากและน่าเบื่อที่สุด และเป็นเหตุผลเดียว
> ที่โปรเจคเราควรมีตัวตน

---

## 2. งานวิจัย — Extraction (กระทบ Phase 2 และ 4)

### 2.1 Layout-Aware Parsing Meets Efficient LLMs (arXiv 2510.09722, Alibaba)

ใกล้ Phase 2 ของเรามากที่สุดในบรรดาทุกชิ้นที่เจอ

- สร้าง **SynthResume** 2,994 ใบ: คัด template ด้วยมือให้ครอบคลุมทั้ง layout คอลัมน์เดียวและหลายคอลัมน์
  → ให้ LLM แทนค่าในฟิลด์โดยคงโครงสร้างเดิม → คนตรวจ → ยิงกลับเข้า template เป็น PDF
- label ได้มาแบบ 2 รอบ: โมเดล pre-label แล้วคนแก้
- ประเมินแยก **Accuracy** (= correct / aligned) ออกจาก **F1** (= correct / predicted, / ground truth)
  เพื่อแยก "จับคู่ entity ผิด" ออกจาก "ดึงค่าในฟิลด์ผิด"
- จับคู่ entity ด้วย **Hungarian algorithm** บน similarity matrix — แก้ปัญหาลำดับและจำนวนไม่ตรงกัน
- กฎเทียบค่าต่างกันตามชนิดฟิลด์: วันที่ normalize เป็น (ปี, เดือน) · ชื่อ entity ใช้ substring
  · ข้อความยาวใช้ edit distance > 0.9 · อีเมล/ชื่อใช้ exact match หลัง lowercase + ตัดวรรคตอน
- ประมาณว่า resume ~20% ใช้ layout ที่อ่านเรียงบรรทัดตรง ๆ ไม่ได้
- ผล F1: parser เชิงพาณิชย์ 0.76–0.82 · Claude-4 ยิง OCR ดิบ 0.92 · ใส่ pipeline layout 0.95–0.96
  · Qwen3-0.6B fine-tune 0.96 บนชุดจริง เร็วกว่า ~3 เท่า
- ฟิลด์ที่ layout ช่วยมากที่สุดคือ **ข้อความยาว** (0.548 → 0.854)

> **🔴 เทคนิคที่ควรขโมยมาทั้งดุ้น:** โมเดลไม่ต้องคายข้อความออกมา ให้คาย **ช่วงเลขบรรทัด**
> แล้ว post-processor ไปดึงข้อความจริงตามเลขบรรทัดนั้น · entity ที่ verify กลับไปที่ต้นฉบับไม่ได้ → ทิ้ง
>
> **เอาไปใช้:** นี่คือวิธีทำให้เกณฑ์ "ทุกทักษะชี้กลับต้นฉบับได้ 100%" (charter §4) เป็นจริง
> *โดยการออกแบบ* ไม่ใช่ด้วยการตรวจทีหลัง และกัน LLM แต่งข้อมูล (charter §8.5) ไปในทางเดียวกัน
> → เขียนลง `01-evaluation-plan.md` §6 และเป็น output contract ของ Phase 4
>
> **เอาไปใช้ (2):** ตาราง normalize ต่อชนิดฟิลด์ของเขาละเอียดกว่า §3.1 ของเรา — เอามาเทียบ
> โดยเฉพาะ "ข้อความยาวใช้ edit distance ไม่ใช่ exact match" ซึ่งเรายังไม่มีกฎรองรับ

### 2.2 ResumeBench — Beyond Human Labels (EMNLP 2025, pp. 31919–31945)

- 2,500 resume สังเคราะห์ · 50 template · 30 สายอาชีพ · 5 ภาษา · เคลมว่าเป็น benchmark
  resume parsing ตัวแรกที่ privacy-compliant
- สร้างด้วย human-in-the-loop แล้ว validate เทียบ resume จริง · เรนเดอร์ HTML → PDF ด้วย pdfkit ⚠️
- ทดสอบ 24 LLM · พบว่า GPT-4o อ่อนเรื่องการ align โครงสร้างข้ามภาษา
  · โมเดลเล็ก scale ไม่สม่ำเสมอ · โมเดลสายโค้ดแพ้โมเดลทั่วไป
- **บังคับ output เป็น JSON ช่วยให้ตรง schema แต่ไม่ได้แก้ความกำกวมเชิงความหมาย**
- ปล่อยให้ใช้เพื่อการวิจัยที่ไม่ใช่เชิงพาณิชย์ ⚠️ (ต้องอ่าน license จริงก่อนแตะ)

> **เอาไปใช้:** ตัวเลข 50 template / 30 สายอาชีพ เป็นหมุดเทียบว่าความหลากหลายระดับไหนถือว่า "พอ"
> (ของเรา §6.2) · และข้อ "JSON ตรง schema ≠ ถูกความหมาย" คือเหตุผลตรง ๆ ว่าทำไม
> Phase 4 ต้องวัดความถูกต้องของค่า ไม่ใช่วัดแค่ว่า parse JSON ผ่าน
>
> **ต้องเช็ก:** ชุดนี้ใช้เป็น Tier C (stress test) ได้หรือไม่ ถ้า license อนุญาต — จะได้ layout
> 50 แบบฟรีโดยไม่ต้องเสี่ยงเรื่องข้อมูลบุคคลเหมือน PDF จาก Kaggle

### 2.3 MSLEF (arXiv 2509.06200)

- ผสม resume จริงจาก Kaggle 2,400 ใบ + สังเคราะห์ด้วยสคริปต์ Python ~1,000 ใบ
- สังเคราะห์เพื่อ **เพิ่มความหลากหลายและเล่นกับ edge case โดยเฉพาะ** ไม่ใช่เพื่อเพิ่มจำนวน

> **เอาไปใช้:** ยืนยันแนวทาง 3 tier ของเรา — สังเคราะห์มีหน้าที่ควบคุม edge case ได้ตามใจ
> ซึ่งข้อมูลจริงให้ไม่ได้ · ใช้อ้างอิงได้เวลาต้องอธิบายว่าทำไมไม่ใช้ข้อมูลจริงเป็นชุดหลัก

### 2.4 PopResume (arXiv 2603.22714)

- 60,884 resume · ใช้ template สาธารณะจาก LiveCareer · เนื้อหาสร้างจาก rule-based
  ที่ขึ้นกับตัวแปรเชิงโครงสร้างเท่านั้น (ไม่ใช้ LLM เขียนเนื้อหา)
- จุดประสงค์คือ **causal fairness evaluation** ของตัวคัดกรอง resume ที่เป็น LLM/VLM

> **เอาไปใช้:** วิธี rule-based ล้วนทำให้ควบคุมตัวแปรได้เป๊ะ จึงทำ counterfactual ได้
> (เปลี่ยนแค่ชื่อ เหลือทุกอย่างเท่าเดิม) — ดูข้อ 6.2

---

## 3. งานวิจัย — Matching / Ranking (กระทบ Phase 3)

### 3.1 ตระกูล ConFit (arXiv 2401.16349 → 2502.12361 → 2605.09760)

- v1: paraphrase ส่วนต่าง ๆ ของ resume/JD เป็น data augmentation + contrastive learning
  · ดีขึ้นกว่าวิธีก่อนหน้าถึง 19% / 31% absolute nDCG@10
- v2 (ACL Findings 2025): hypothetical resume embedding + runner-up hard-negative mining
  · ดีขึ้นเฉลี่ย recall 13.8% / nDCG 17.5% ⚠️
- v3: เพิ่ม LLM re-ranking ทับชั้น retrieval · ดีขึ้นอีก 7.81% ⚠️
- **ใช้ nDCG@10 และ recall เป็น metric หลัก** ตรงกับที่เราเลือกไว้ใน §4.3
- 🔴 **ผู้เขียนระบุว่าไม่มีชุดข้อมูล person-job fit มาตรฐานที่เปิดสาธารณะ** ชุด AliYun
  จากการแข่งปี 2019 ถูกถอดไปแล้ว · ชุดที่เหลือเป็นข้อมูลบริษัท (resume–job pairs ที่มี label รับ/ไม่รับ)

> **เอาไปใช้ (สำคัญ):** ข้อสุดท้ายคือ **เหตุผลภายนอกที่หนักที่สุด** ที่รองรับการตัดสินใจของเรา
> ว่าต้อง annotate ranking เอง — ไม่ใช่เพราะเราขี้เกียจหา แต่เพราะมันไม่มีให้หา
> → ใส่เป็นเชิงอรรถใน `02-dataset-strategy.md` §1
>
> **เอาไปใช้ (2):** โครง v1→v2→v3 คือบันได baseline ที่เขาไต่จริง: embedding →
> ปรับการเทรน → LLM re-rank · เทียบกับบันไดของเรา (§4.4) ว่าเราข้ามขั้นไหนไปหรือเปล่า
>
> **ข้อควรระวัง:** ตัวเลข 19–31% ของเขาวัดบนข้อมูลหลายหมื่นคู่ · ของเรามี 6 ตำแหน่ง
> ส่วนต่างระดับนั้นเราจะไม่เห็นและไม่ควรคาดหวัง

### 3.2 Resume2Vec (Electronics 2025, MDPI)

- รายงาน nDCG 0.87 เทียบกับ ATS แบบเดิม 0.83 · ตัดสินโดยเทียบกับความชอบของมนุษย์ ⚠️
- **ไม่รายงานเพดานมนุษย์ และไม่รายงาน confidence interval**

> **เอาไปใช้:** เป็นตัวอย่างของสิ่งที่เราตั้งใจไม่ทำ — 0.87 vs 0.83 ไม่มี CI แปลว่าบอกไม่ได้ว่าต่างจริงหรือไม่
> ใช้เป็นเหตุผลประกอบ §5.1 (bootstrap CI) ได้ดี

### 3.3 JobResQA (arXiv 2601.23183, Avature) — มี repo ที่ `Avature/jobresqa-benchmark`

ไม่ใช่งาน matching แต่เป็น **reading comprehension บน resume + JD** ซึ่งใกล้ Phase 4 ของเรา

- 581 QA pairs บน 105 คู่ resume–JD · 24 อุตสาหกรรม · 194 ตำแหน่ง
- เก็บ resume/JD จริงจาก job board → de-identify → ให้ GPT-4.1 เขียนใหม่เป็นฉบับสังเคราะห์
- 5 ภาษา (อังกฤษ/สเปน/อิตาลี/เยอรมัน/จีน) แปลแบบ human-in-the-loop
- คำถามแบ่ง 3 ระดับ: basic 26.5% / intermediate 36.7% / **complex 36.8%**
  (complex = ต้องให้เหตุผลข้ามเอกสาร)
- **คำตอบทุกข้อมีทั้งคำตอบสั้นและคำอธิบาย และคำอธิบายต้องอ้างรายละเอียดจาก resume/JD**
- ให้คะแนนด้วย G-Eval (ใช้ Claude Sonnet 4 เป็น judge) สเกล 0–1 เทียบกับคำตอบอ้างอิงที่คนเขียน
- ผล: Llama 3.3 70B ได้ 0.73 (อังกฤษ) / 0.69 (สเปน) แต่ตกลงเหลือ 0.47 (เยอรมัน) / 0.48 (จีน)
  · SD สูงถึง 0.39 · โมเดล 1B ได้ 0.15–0.35

> **เอาไปใช้:** (ก) วิธี de-identify แล้วให้ LLM เขียนใหม่ เป็นทางสายกลางระหว่าง "สังเคราะห์ล้วน"
> กับ "ใช้ของจริง" ที่เรายังไม่ได้พิจารณาใน `02` §2 — ได้ layout และสำนวนจริงโดยไม่พาข้อมูลบุคคลมา
> (ข) การบังคับว่าคำอธิบายต้องอ้างกลับเอกสาร คือหลักเดียวกับข้อ 2.1 มาจากคนละทีม
> (ค) ตัวเลข SD 0.39 เตือนว่า LLM-as-judge แปรปรวนสูง — ถ้า Phase 4 จะใช้ ต้องรันหลายครั้งแล้วรายงาน SD
> (ง) ช่องว่างข้ามภาษาใหญ่มาก → ยืนยัน charter §5 ว่าเริ่มด้วยอังกฤษถูกแล้ว

---

## 4. งานวิจัย — Skill Extraction และ Taxonomy (กระทบ `dataset/taxonomy/skills.v0.yaml`)

### 4.1 SkillSpan (NAACL 2022) และสายต่อ ๆ มา

- span-level hard/soft skill extraction บน job posting ~14.5K ประโยค
- ตัวเลขที่รายงาน (คนละ setup เทียบกันตรง ๆ ไม่ได้ ⚠️): LLaMA 3 8B fine-tune ได้ Span F1 64.8%
  · **GPT-4 แบบ prompt เปล่า ได้แค่ 27.8%** · weak supervision + ESCO embedding ได้
  strict F1 49.44 / loose F1 74.41
- 🔴 **แทบทั้งหมดเป็น job posting ไม่ใช่ resume** — ไม่มี benchmark skill extraction บน resume ที่เป็นมาตรฐาน
- SkillSpan-ESCO มีข้อจำกัดที่รู้กัน: map ทักษะหนึ่งไปยัง ESCO ตัวแทนตัวเดียว
  แม้ความจริงจะตรงหลายตัว → Skill-XL ทำ label ให้ครบถ้วนกว่า
- Chinese-SkillSpan (arXiv 2604.23009) รายงาน 3 ตัว: **Exact-span F1 · Relaxed F1 (IoU ≥ 0.5)
  · Concept Accuracy**

> **🔴 เอาไปใช้ (กระทบเป้าใน charter โดยตรง):** strict F1 ของ state of the art บนงาน
> span-level skill extraction อยู่ที่ราว 0.49–0.65 ส่วนเป้าของเราคือ micro F1 ≥ 0.80 / macro ≥ 0.65
> ตัวเลขนี้ **ไม่ขัดกันโดยตรง** เพราะงานของเราง่ายกว่าหลายอย่าง (แค่ตัดสินว่ามีทักษะนี้หรือไม่
> ไม่ต้องหาขอบ span ให้เป๊ะ · taxonomy ปิดและเราเขียนเอง · ข้อมูลสังเคราะห์ที่เราคุมได้)
> แต่ถ้าวัดแบบ strict span แล้วได้ 0.80 ต้องตั้งข้อสงสัยกับตัวเองว่า dataset ง่ายเกินไปไหม
> → บันทึกข้อนี้ไว้เป็นเกณฑ์ sanity check ใน §7 error analysis
>
> **เอาไปใช้ (2):** การรายงาน 3 ตัวแบบ Chinese-SkillSpan (exact / relaxed IoU / concept)
> เหมาะกับเรามาก เพราะเรามีทั้งเกณฑ์ "เจอทักษะ" และเกณฑ์ "ชี้ span กลับได้" ซึ่งเป็นสองเรื่อง
> → เสนอให้แยก metric สองตัวนี้ออกจากกันใน §3.2
>
> **เอาไปใช้ (3):** ESCO มีอยู่จริงและเป็นมาตรฐานที่คนใช้อ้าง · ก่อนจะขยาย `skills.v0.yaml`
> ด้วยมือต่อ ควรดูว่า map canonical id ของเราเข้า ESCO ได้ไหม — ได้ทั้งความน่าเชื่อถือ
> และทางออกจากปัญหา "คำย่อกำกวม" (§4.4 ของ annotation guidelines)

---

## 5. Portfolio Projects บน GitHub — คู่เทียบที่ตรงที่สุด

มีโปรเจคที่ "ตรงแนวเรา" จำนวนมาก (GitHub topic `resume-matcher` มีราว 45 repo)
หลายอันเขียน tagline แบบเดียวกับ charter ของเราเลย — "evidence-backed", "decision support only,
no auto-rejection", "never invents experience"

### 5.1 `maxprogerit/resumeScreen` — ใกล้ที่สุด (อ่าน README แล้ว)

| ด้าน | เขาทำอย่างไร |
|---|---|
| Stack | FastAPI + HTML/JS static · SQLAlchemy + Alembic · SQLite (dev) / PostgreSQL (docker) |
| PDF | รองรับ PDF + DOCX · PDF สแกนใช้ RapidOCR (ONNX) · **ตัด hidden text ออกก่อนสกัด** |
| บทบาท LLM | ใช้แค่ 3 อย่าง: สกัดโปรไฟล์, สกัด requirement, เขียนคำอธิบาย · Pydantic บังคับ schema |
| การให้คะแนน | **คำนวณด้วย Python ล้วน ไม่ใช่ LLM** · 5 องค์ประกอบ 0–1 (must-have, nice-to-have, ประสบการณ์จากวันที่, semantic similarity, ภาษา/การศึกษา) · weighted average เฉพาะองค์ประกอบที่ job นั้นมี · เปลี่ยน weight แล้วคิดใหม่ได้โดยไม่เรียก LLM ซ้ำ |
| หลักฐาน | เก็บ quote ที่ LLM เสนอ **เฉพาะที่ปรากฏตรงตัวใน resume** หลัง normalize ช่องว่าง · quote ที่แต่งขึ้น → ทิ้งและนับไว้ |
| ลูกเล่นที่ดี | ทักษะที่โผล่แค่ในรายการ skills เปล่า ๆ จะติดธง **"listed only"** — กัน keyword stuffing แต่ไม่หักคะแนน |
| Fairness | มีสคริปต์ counterfactual audit: resume ที่ต่างกันแค่ชื่อ/คำนำหน้า/สรรพนาม ต้องได้คะแนนเท่ากันเป๊ะ |
| ข้อมูล | สังเคราะห์ทั้งหมด · มี generator สร้าง PDF/DOCX + seed 1 job 25 ผู้สมัคร |
| **🔴 สิ่งที่ไม่มี** | **ไม่รายงาน accuracy / precision / ranking quality เลย** · test suite ใช้ fake LLM ตรวจแค่ว่าโค้ดทำงาน ไม่ได้ตรวจว่าผลถูก |

> **เอาไปใช้ — นี่คือข้อสรุปที่สำคัญที่สุดของเอกสารนี้:**
> โปรเจคนี้มีทุกอย่างที่เรากำลังจะสร้าง (FastAPI, หลักฐาน verbatim, คะแนนแบบ deterministic,
> ข้อมูลสังเคราะห์, docker, fairness audit) และอาจเสร็จเร็วกว่าเราหลายเท่า
> **แต่ตอบคำถาม "รู้ได้ยังไงว่ามันแม่น" ไม่ได้** — ไม่มีชุดเฉลย ไม่มีตัวเลข
>
> → ช่องว่างนั้นคือสิ่งที่ `01`–`03` ของเราทำอยู่ และคือสิ่งเดียวที่ทำให้โปรเจคนี้ต่าง
> ยืนยันว่าลำดับ data-first (charter §7) ถูกแล้ว **อย่าลดความสำคัญของ Phase 1 เพื่อรีบไปทำ UI**
>
> → ขณะเดียวกัน เขาคิดได้หลายอย่างที่เรายังไม่มีในแผน: ตัด hidden text, ธง "listed only",
> แยกการให้คะแนนออกจาก LLM 100% เพื่อปรับ weight ได้โดยไม่ต้องเรียกใหม่ (ดูข้อ 6)

### 5.2 อันอื่นที่น่าดูต่อ (ยังไม่ได้เปิดอ่าน)

| repo | จุดที่น่าสนใจ |
|---|---|
| `dileepreddy27/careermatch-engine` | chunk แบบรู้ section พร้อม **exact source offsets** · FastAPI + Qdrant · มี synthetic demo ที่ reproduce ได้ |
| `krtk06/ResumeAnalysis` | อ้างบรรทัดเป๊ะ ๆ ต่อทุก match โดย verify กับข้อความที่สกัดได้ · มี calibrated local scorer |
| `rahulmurarisetty/resume-evidence-matcher` | มีสถานะ **"adjacent"** = ไม่มีหลักฐานตรง แต่มี bullet ที่เกี่ยวข้อง |

> **เอาไปใช้:** สถานะ "adjacent" ของ repo ที่สามคือสิ่งที่ `03-annotation-guidelines.md` §4.2
> ของเราห้ามไว้ (ห้ามใส่ทักษะที่ได้จากการอนุมาน) — เรายังยืนยันจุดยืนเดิม แต่ควรอ่านว่าเขานิยามไว้ยังไง
> เพราะในการ *แสดงผล* เราอาจต้องมีหมวดนี้ ถึงจะไม่นับมันในการวัดผล

### 5.3 อีก 5 repo — ผลตรวจ (2026-11-18)

ชุดนี้มาจากการค้นหาแยกอีกรอบ · **ตรวจ README ทุกอันแล้ว มีจริงทั้ง 5 repo**
แต่คำบรรยายที่ได้มาตอนแรกเพี้ยนหลายจุด บันทึกไว้ด้านล่าง

| repo | ขนาด | ของจริงเป็นอย่างไร |
|---|---|---|
| `Sharawey74/Recruiter-Pro` | 140 commits · 1 ⭐ | Next.js 16 + FastAPI + scikit-learn · Ollama (default) / OpenRouter / fallback เป็น rule-based |
| `clawnify/OpenRecruit` | 9 commits · 1 ⭐ | **TypeScript/Vite/Cloudflare ไม่ใช่ Python** · เป็น app template ของแพลตฟอร์ม Clawnify ไม่ใช่โปรเจคเดี่ยว |
| `erAnjaliGoyal/Resume-Screener-LLM` | **2 commits** · 0 ⭐ | Python + FastAPI + Claude API · มี CLI + API + Docker |
| `adiagrawaal/AI-Resume-Screening-System` | **3 commits** · 0 ⭐ | Groq + Llama 3.3 70B + Pydantic + pypdf/python-docx · uv |
| `suraj07nangare/ai-resume-screening-system` | 10 commits · 1 ⭐ | FastAPI + Pydantic v2 + SQLAlchemy 2 + Alembic + PostgreSQL/SQLite + Streamlit + Docker |

> **⚠️ ข้อควรระวังเรื่องน้ำหนักของแหล่ง:** 3 ใน 5 อันมี commit 2–3 ครั้งและ 0 ดาว
> Recruiter-Pro ที่ดูใหญ่สุดก็มี README ที่ตัวเลขขัดกันเอง (test 544 กับ 530/529 · coverage
> 84.07% กับ 83.7%) · ทั้งหมดนี้คือ portfolio project ที่น่าจะเขียนด้วย AI ช่วย
> **ใช้เป็นแหล่งไอเดียการออกแบบได้ ใช้อ้างเป็นหลักวิชาไม่ได้** · และภาพประกอบที่แนบมากับลิสต์
> เป็นภาพโฆษณาของผลิตภัณฑ์เชิงพาณิชย์คนละตัว (Land Surveyors United, Codester, GroveTech,
> Keka, Blink) ไม่ใช่ screenshot ของ repo เหล่านี้

#### 5.3.1 `Recruiter-Pro` — 🔴 ทำคนละโจทย์กับเรา

- **ทิศทางกลับกัน:** รับ resume **1 ใบ** แล้วให้คะแนนเทียบกับคลังตำแหน่งงาน **800 ตำแหน่ง**
  → เป็น *job recommendation ให้ผู้สมัคร* ไม่ใช่ *จัดอันดับผู้สมัครให้ตำแหน่ง* แบบเรา
  (คำบรรยายที่ได้มาตอนแรกบอกว่าจัดอันดับผู้สมัคร — ไม่ถูก)
- "AI Agent 4 ตัว" (Parser / Extractor / Scorer / Explainer) **เป็นฟังก์ชันที่เรียกกันใน
  process เดียว ไม่ใช่ agent ที่คุยข้ามเครือข่าย** — อย่าหลงคิดว่าต้องทำสถาปัตยกรรม multi-agent
- มี badge LangChain ใน README แต่ไม่อธิบายว่าใช้ทำอะไร ⚠️
- **skill normalization: 679 canonical skills + 1,554 aliases** (React / ReactJS / React.js → ตัวเดียว)
- น้ำหนักที่ประกาศไว้: skill 50% · experience 20% · title 17% · education 8% · keywords 5%
  แล้วผสมกับคะแนน ML แบบ 60/40
- เทรน ML จาก `data/AI_Resume_Screening.csv` ซึ่งเป็นชุดสังเคราะห์สาธารณะ ⚠️
- รายงาน latency: rule scoring ~0.7 วินาทีสำหรับ 800 ตำแหน่ง · ~1.8 วินาทีเมื่อเปิดโมเดล
  · ~4.5 วินาที end-to-end จาก PDF จริง
- README ระบุเองว่า **ยังไม่ได้ประเมิน fairness หรือ adverse impact** และห้ามใช้ตัดสินใจจ้างจริง

> **เอาไปใช้:** (ก) 679/1,554 เทียบกับ `skills.v0.yaml` ของเราที่มี **104 ทักษะ** — เป็นหมุดบอก
> ว่าถ้าจะครอบคลุมจริงต้องใหญ่อีกราว 6 เท่า · แต่ taxonomy เล็กที่ *เรา annotate ครบ* มีค่ากว่า
> taxonomy ใหญ่ที่ไม่มีเฉลย ดังนั้นยังไม่ต้องรีบขยาย (ย้ำ `03` §4.3)
> (ข) latency 0.7 วินาทีต่อ 800 ตำแหน่งแบบ rule-based ยืนยันว่าเป้า < 2 วินาทีของเรา (charter §4)
> ตั้งไว้สมเหตุสมผล ไม่ตึงเกิน
> (ค) **ไม่มีใครในทั้ง 9 repo อธิบายที่มาของน้ำหนักด้วยข้อมูล** — skill 50% มาจากไหน? ไม่มีคำตอบ
> นี่คือช่องว่างที่ Phase 3 ของเราเติม เพราะเราจะเลือกจากผลเปรียบเทียบ ไม่ใช่เลือกจากความรู้สึก

#### 5.3.2 `OpenRecruit` — 🔴 คนที่ 3 ที่คิดเรื่องหลักฐานตรงกับเรา

- AI อ่าน CV เทียบกับ requirement ของตำแหน่ง **และบังคับว่าต้องแนบหลักฐาน**
- ระบบไปเช็กว่าข้อความที่ยกมาปรากฏอยู่ในข้อความ CV ที่สกัดได้จริงหรือไม่
  · **ข้อที่ match ไม่ได้ → ติดป้าย "unverified" และตัดออกจากการคิดคะแนน**
- README ระบุว่า AI **ไม่มีสิทธิ์** advance / reject / hire — มนุษย์เท่านั้น
- มี seed dataset ที่จงใจใส่เคส unverified ไว้ให้เห็น · มี test สำหรับกฎการ verify (`pnpm test`)
- ฟีเจอร์ fairness: ซ่อนชื่อผู้สมัครในด่านแรก ๆ (ปิดได้)
  · **ซ่อนผลประเมินของเพื่อนร่วมทีมจนกว่าเราจะส่งของตัวเองก่อน**
- ฝั่ง workflow ครบจริง: careers site + JobPosting structured data + job feed + kanban ต่อตำแหน่ง
  + stage-time alert + เหตุผลการตัดออก + talent pool + รายงาน time-to-hire

> **🔴 เอาไปใช้ (สำคัญ):** ข้อ "ซ่อนผลประเมินของคนอื่นจนกว่าจะส่งของตัวเอง" คือ
> **หลักกัน anchoring bias ตัวเดียวกับ `03-annotation-guidelines.md` §4.5** (ห้ามใช้
> pre-annotation กับ Tier B) มาจากคนละบริบทโดยสิ้นเชิง — ยืนยันว่ากฎข้อนั้นของเราไม่ได้คิดมากไปเอง
>
> **เอาไปใช้ (2):** การตัดหลักฐานที่ verify ไม่ได้ **ออกจากคะแนน** (ไม่ใช่แค่ไม่แสดง) ชัดกว่าที่
> charter เราเขียนไว้ · ของเราบอกแค่ว่าต้องชี้กลับได้ 100% แต่ไม่ได้บอกว่าถ้าชี้ไม่ได้แล้วคะแนนเป็นอย่างไร
> → ต้องเติมกฎนี้ให้ชัดใน `01` §6 และเป็น output contract ของ Phase 4
>
> **เอาไปใช้ (3):** ฝั่ง workflow เป็นสิ่งที่ charter §2.3 ตัดออกไปแล้วโดยตั้งใจ — ดูไว้เพื่อยืนยันว่า
> ถ้าทำ จะบานปลายแค่ไหน **ไม่ใช่เพื่อเอามาทำ**

#### 5.3.3 `Resume-Screener-LLM` — มีชุดตัวอย่างที่ใกล้ "เฉลย" ที่สุดในกลุ่มนี้

- Python + FastAPI + **Claude API** · มี CLI และ API · Docker + Codespaces
- Claude ให้คะแนนตาม **rubric ที่เขียนเป็น JSON**
- output ต่อผู้สมัคร: ทักษะที่ตรง / ที่ขาด / จุดแข็งที่ถ่ายโอนได้ / red flags / สรุป
- `examples/` มี JD 1 ใบ + **resume สมมติ 6 ใบ (PDF, DOCX, TXT) ที่ตั้งใจให้ผลเรียงลำดับออกมาแบบที่รู้อยู่แล้ว**
- ไม่มี metric · บอกว่า bias audit เป็นงานในอนาคต

> **เอาไปใช้:** "resume 6 ใบที่รู้ลำดับคำตอบอยู่แล้ว" คือ **smoke test ที่เราควรมีด้วย** และ
> ทำได้ *ก่อน* dataset เต็มเสร็จ — ใช้ตรวจว่า pipeline ไม่พังระหว่างพัฒนา โดยไม่ต้องเปิด lockbox
> (`01` §8.1) · ต่างจากชุดวัดผลจริงชัดเจน ให้เรียกมันว่า smoke set และห้ามรายงานตัวเลขจากมัน
>
> **เอาไปใช้ (2):** การเขียน rubric เป็น JSON แล้วยัดเข้า prompt เป็นแนวที่ตรงกับ `03` §6.2
> ที่เรามี rubric เกรด 0–3 อยู่แล้ว — Phase 4 แปลง rubric ของเราเป็น JSON ได้ตรง ๆ

#### 5.3.4 `AI-Resume-Screening-System` — ตัวอย่างการใช้ free tier

- **Groq + Llama 3.3 70B Versatile** · Pydantic + pypdf + python-docx · Python 3.13 + uv
- LLM parse JD เป็น requirement มีโครงสร้าง แล้ว parse resume แล้วเทียบ
- สกัด: ชื่อ อีเมล เบอร์ ทักษะ ประสบการณ์ การศึกษา โปรเจค ใบรับรอง
- **ไม่มีชุดทดสอบเลย** — ตัดออกเพราะความเป็นส่วนตัว · README บอกเองว่า
  ตัวเลขที่โชว์ขึ้นกับ resume, JD และคำตอบของ LLM ในรอบนั้น
- ระบุว่าเป็นโปรเจคเพื่อการเรียนและ portfolio

> **เอาไปใช้:** charter §6 ทิ้งไว้ว่า LLM provider ยังไม่สรุป และ Groq free tier เป็นตัวเลือกหนึ่ง
> — repo นี้คือหลักฐานว่า Llama 3.3 70B บน Groq ใช้ทำงานนี้ได้จริง เทียบกับ JobResQA (ข้อ 3.3)
> ที่วัด Llama 3.3 70B ได้ 0.73 บนภาษาอังกฤษ **ซึ่งเป็นตัวเลขที่เราพอจะคาดหวังได้**
> → บันทึกเป็นตัวเลือกตั้งต้นของ Phase 4 พร้อมเลข 0.73 เป็นเส้นอ้างอิง
>
> **และเป็นตัวอย่างที่ชัดที่สุดว่าทำไมเราต้องมี dataset:** โปรเจคนี้ตอบไม่ได้แม้แต่ว่า
> รันสองครั้งจะได้ผลเหมือนกันไหม

#### 5.3.5 `ai-resume-screening-system` (suraj) — สมบูรณ์ที่สุดเชิงวิศวกรรม

- stack ใกล้แผนเราที่สุด: FastAPI + Pydantic v2 + SQLAlchemy 2 + Alembic + PostgreSQL
  (SQLite สำหรับ dev/test) + Streamlit + docker-compose
- **LLM layer แบบ provider-agnostic: Gemini / OpenAI / Anthropic / Groq / OpenRouter / Ollama
  + mock provider**
- **OCR fallback ที่มีเกณฑ์เป็นตัวเลข: ลอง pdfplumber ก่อน ถ้าข้อความที่ได้ < 50 ตัวอักษร
  → ไป pytesseract + pdf2image (poppler)**
- คะแนนเป็น **deterministic weighted** คำนวณในโค้ด (skills / experience / education / other)
  · LLM ใช้แค่สกัดข้อมูลมีโครงสร้าง
- Ranking Dashboard ต่อตำแหน่ง + endpoint rankings
- test 35 ตัว **mock ทั้งหมด ไม่ต้องมี API key** · fixture PDF สร้างด้วย **reportlab**
  · เทส OCR ใช้ PDF ที่เป็นภาพล้วนที่ generate ขึ้น
- ⚠️ **ส่งอีเมลอัตโนมัติตามเกณฑ์คะแนน** (shortlist / rejection / under review)
  และแนบลิงก์จอง Cal.com เมื่อคะแนน ≥ 80 (ปิดเป็นค่าเริ่มต้น)

> **🔴 เอาไปใช้ (ข้อห้าม):** การส่ง **rejection email อัตโนมัติตามคะแนน AI** คือสิ่งที่
> charter §8.3 ของเราห้ามไว้ตรง ๆ และ §2.3 ตัด "ส่งอีเมลหาผู้สมัคร" ออกจาก scope ไปแล้ว
> · repo นี้มีประโยชน์มากในฐานะ **ตัวอย่างรูปธรรมว่าเส้นแบ่งอยู่ตรงไหน** — ทุกอย่างก่อนหน้านั้น
> (คะแนน deterministic, dashboard, leaderboard) เราทำ · การตัดสินใจแทนมนุษย์เราไม่ทำ
> → เอาเคสนี้ไปเขียนเป็นตัวอย่างใน charter §8 ได้เลย เพราะตอนนี้ §8 เป็นหลักการลอย ๆ ไม่มีตัวอย่าง
>
> **เอาไปใช้ (2):** ดูข้อ 6.5 — repo นี้มีเทคนิคที่ขโมยมาใช้ได้ทันที 3 อย่าง

---

## 6. 🔴 สิ่งที่เราพลาด (รายการที่ต้องลงมือ)

> **✅ สถานะ 2026-11-18: ลงมือแล้วทั้ง 4 ข้อ + เทคนิคย่อยในข้อ 6.5**
>
> | ข้อ | ลงไปที่ไหน |
> |---|---|
> | 6.1 metric แยก span | `01` §3.2 (concept / exact-span / relaxed IoU≥0.5) + §3.3 + §10 ข้อ 5 |
> | 6.2 fairness metric | `01` §11 (ทั้งหมดใหม่) + `02` §6.1 (counterfactual pair ใน generator) + `02` §8 (คอลัมน์ manifest) + charter §10 |
> | 6.3 เพดานมนุษย์ | `01` §5.2 (เปลี่ยนชื่อค่า + หลักฐานภายนอก) + `03` §11.1 (ถ้าต่ำให้แก้ rubric) |
> | 6.4 ปรับ weight ไม่เรียก LLM ใหม่ | `01` §12 (ข้อบังคับสถาปัตยกรรม 3 ข้อ) + charter §10 |
> | 6.5 เทคนิคย่อย 5 อย่าง | `01` §3.1, §3.2 (evidence contract) + `02` §6.1, §6.2, §6.4 + `03` §8.1, §8.3 |
>
> รายละเอียดการแก้ทั้งหมดอยู่ในบันทึกการแก้ไขของแต่ละไฟล์:
> `01` §13 · `02` §11 · `03` §12 · charter §10

### 6.1 Metric ที่ควรแยกเป็นสองตัว แต่เรารวมเป็นตัวเดียว

เป้าใน charter §4 มีสองเรื่องคนละเรื่องปนกัน: "สกัดทักษะได้ถูกต้อง" กับ "ชี้ span กลับไปต้นฉบับได้"
งานวิจัยแยกกันชัด (exact-span F1 / relaxed F1 IoU≥0.5 / concept accuracy)

**ลงมือ:** แก้ `01-evaluation-plan.md` §3.2 ให้รายงานแยก และนิยาม relaxed match ให้ชัดว่าใช้ IoU เท่าไร

### 6.2 เราไม่มี metric สำหรับ fairness เลย

charter §8.2 ห้ามใช้ลักษณะที่ได้รับการคุ้มครองเป็นเกณฑ์ แต่เราไม่มีวิธี *วัด* ว่าทำตามจริงไหม
ทั้ง PopResume (งานวิจัย) และ `resumeScreen` (portfolio) ใช้วิธีเดียวกันคือ **counterfactual**:
สร้าง resume คู่ที่ต่างกันแค่ชื่อ/สรรพนาม แล้วเช็กว่าผลต้องเหมือนกัน
งานปี 2026 ยังเตือนเพิ่มสองเรื่อง: อคติไปซ่อนอยู่ใน **ภาษาที่ใช้ประเมิน** (คำอธิบาย) ไม่ใช่แค่คะแนน ⚠️
และผลการ audit ขึ้นกับวิธี audit เองไม่น้อยกว่าขึ้นกับตัวโมเดล ⚠️

**ลงมือ:** generator ของ Phase 1 ควรสร้าง **counterfactual pair** ตั้งแต่แรก (โปรไฟล์เดียวกัน
เปลี่ยนแค่ชื่อ) เพราะการมาทำย้อนหลังแพงกว่ามาก · เพิ่มเป้าข้อใหม่: "คะแนนของคู่ counterfactual
ต้องเท่ากันเป๊ะ" — ซึ่งเป็นเป้าที่ผ่านง่ายถ้าคะแนนเป็น deterministic และเป็นเป้าที่ *มีความหมาย*
ทันทีที่ LLM เข้ามาเกี่ยวใน Phase 4

### 6.3 เพดานมนุษย์อาจต่ำกว่าที่เราคิดมาก

เราวางแผนวัดเพดานมนุษย์ด้วย test–retest กับตัวเอง (`03` §11) ซึ่งถูกทาง
แต่ตัวเลขจากที่อื่นน่ากังวล: งานหนึ่งที่ให้คน 3 คน annotate งานคัดกรอง resume ได้ Fleiss' kappa
เพียง 0.167 (binary) / 0.122 (3 คลาส) จนผู้เขียนปฏิเสธที่จะถือว่า annotator คนใดคนหนึ่งเป็น
ground truth ⚠️ · ฝั่ง IR ก็มีรายงานว่าผู้ตรวจสองคนให้คะแนน relevance สเกล 0–4 ตรงกันเป๊ะเพียง 42%

**ลงมือ:** เรามี annotator คนเดียว ซึ่ง *ดี* ต่อความสม่ำเสมอ (test–retest วัดได้)
แต่ปิดจุดอ่อนไม่ได้ — ให้เขียนข้อจำกัดนี้ลง `01` §5.2 ตรง ๆ และถ้าผล test–retest ของ ranking
ออกมาต่ำ ให้ถือว่าปัญหาอยู่ที่ **rubric** (`03` §6.2) ไม่ใช่ที่ตัวคน แล้วแก้ rubric ก่อนจะ annotate ต่อ

### 6.4 เรายังไม่ได้ออกแบบให้ "ปรับ weight ได้โดยไม่เรียก LLM ใหม่"

`resumeScreen` แยกชัดว่า LLM ทำแค่สกัดและอธิบาย ส่วนการให้คะแนนเป็น Python ล้วน
ผลคือปรับ weight แล้วคิดคะแนนใหม่ได้ทันที — สำคัญมากสำหรับเรา เพราะ Phase 3 ต้องเทียบวิธี
matching หลายแบบบน dataset เดิมซ้ำ ๆ และเราไม่มีงบ LLM (charter §6)

**ลงมือ:** กำหนดเป็นข้อบังคับทางสถาปัตยกรรมตั้งแต่ตอนนี้: **ผลการสกัดต้อง cache เป็น JSON
บนดิสก์ และ matching ต้องเป็นฟังก์ชันบริสุทธิ์จาก (resume JSON, job YAML) → คะแนน + หลักฐาน**
ไม่เรียก LLM ไม่แตะ I/O · จะทำให้ Phase 3 รันซ้ำได้ไม่จำกัดและ test ได้ง่าย

> **ยืนยันจาก 2 repo โดยอิสระ:** `resumeScreen` ใช้ fake LLM ใน test ส่วน
> `suraj07nangare/...` มี **mock provider** อยู่ใน provider layer เลย และ test 35 ตัว
> รันได้โดยไม่ต้องมี API key · ตรงกับ charter §6 ที่ตั้งใจ "ออกแบบเป็น interface ก่อน
> แล้วเสียบ provider ทีหลัง" → **mock provider ต้องเป็น provider ตัวแรกที่เขียน ไม่ใช่ตัวสุดท้าย**

### 6.5 เทคนิคย่อยที่ขโมยมาใช้ได้ทันที

| เทคนิค | มาจาก | เอาไปใช้ตรงไหน |
|---|---|---|
| **เกณฑ์ OCR fallback เป็นตัวเลข** — ลอง pdfplumber ก่อน ถ้าได้ข้อความ < 50 ตัวอักษร → OCR | 5.3.5 | Phase 2 · เรายังไม่มีกฎว่าจะรู้ได้ยังไงว่า PDF ใบนี้สกัดไม่ออก |
| **สร้าง PDF fixture ด้วย reportlab** (รวมถึง PDF ที่เป็นภาพล้วนเพื่อเทส OCR) | 5.3.5 | Phase 1 · `02` §6.1 ยังไม่ได้ระบุว่าจะเรนเดอร์ PDF ด้วยอะไร · ResumeBench ใช้ pdfkit (HTML→PDF) เป็นอีกทางเลือก |
| **ตัด hidden text ออกก่อนสกัด** | 5.1 | Phase 2 · กันเคสคนฝังคำใน PDF แบบมองไม่เห็น — เป็น adversarial case ที่เรายังไม่ได้คิดถึง |
| **ธง "listed only"** — ทักษะที่โผล่แค่ในรายการ skills เปล่า ๆ ติดธงไว้ แต่ไม่หักคะแนน | 5.1 | ตรงกับ `03` §4.3 ที่เราแยก "ประกาศไว้" กับ "มีหลักฐานการใช้งาน" อยู่แล้ว → ทำให้เป็น field ใน output schema |
| **สถานะ "unverified" ที่ตัดออกจากคะแนน** | 5.3.2 | `01` §6 · นิยามให้ชัดว่าหลักฐานที่ verify ไม่ได้ ไม่ใช่แค่ไม่แสดง แต่ไม่นับ |

---

## 7. ที่ยังไม่ได้หา

- **license ของ ResumeBench / SynthResume** ว่าใช้เป็น Tier C ได้ไหม
- repo 3 อันในข้อ 5.2 ยังไม่ได้เปิดอ่านโค้ด · ข้อ 5.3 อ่านแค่ README ยังไม่ได้เปิดโค้ดอันไหน
- การเทียบ **pdfplumber vs PyMuPDF** แบบมีตัวเลข (ของเราจะตัดสินใน Phase 2) —
  ยังไม่ได้หาว่ามีใครเทียบไว้แล้วบ้าง · ข้อมูลที่มีตอนนี้มีแค่ว่า `suraj07nangare/...`
  เลือก pdfplumber เป็นตัวหลัก **ซึ่งเป็นการเลือกที่ไม่ได้วัด** ใช้อ้างไม่ได้
- ชุด `AI_Resume_Screening.csv` (ที่ Recruiter-Pro ใช้เทรน) — เป็นข้อมูลตารางสังเคราะห์
  ยังไม่รู้ว่ามี PDF จริงหรือไม่ และ license เป็นอะไร
- **ESCO**: โครงสร้างจริง, ขนาด, ภาษาที่รองรับ, map เข้า `skills.v0.yaml` ได้แค่ไหน
- งานที่วัดว่า **layout diversity สำคัญแค่ไหน** — ยังไม่เจอใครวัดตรง ๆ
- ผล full-text ของเปเปอร์ที่ติด ⚠️ ทุกใบ

---

## 8. แหล่งอ้างอิง

**Extraction**
- [Layout-Aware Parsing Meets Efficient LLMs (arXiv 2510.09722)](https://arxiv.org/html/2510.09722v1)
- [ResumeBench — Beyond Human Labels (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1626) · [วิดีโอนำเสนอ](https://underline.io/lecture/131380-beyond-human-labels-a-multi-linguistic-auto-generated-benchmark-for-evaluating-large-language-models-on-resume-parsing)
- [MSLEF (arXiv 2509.06200)](https://arxiv.org/html/2509.06200v1)
- [Evaluating LLM-Based Resume Information Extraction (SCITEPRESS 2025)](https://www.scitepress.org/Papers/2025/138379/138379.pdf)

**Matching / Ranking**
- [ConFit v1 (arXiv 2401.16349)](https://arxiv.org/pdf/2401.16349) · [v2 (ACL Findings 2025)](https://aclanthology.org/2025.findings-acl.661.pdf) · [v3 (arXiv 2605.09760)](https://arxiv.org/html/2605.09760v1)
- [JobResQA (arXiv 2601.23183)](https://arxiv.org/html/2601.23183v1) · repo: `github.com/Avature/jobresqa-benchmark`
- [Resume2Vec (Electronics 2025)](https://www.mdpi.com/2079-9292/14/4/794)

**Skill Extraction / Taxonomy**
- [SkillSpan (NAACL 2022)](https://www.researchgate.net/publication/362270563_SkillSpan_Hard_and_Soft_Skill_Extraction_from_English_Job_Postings)
- [Chinese-SkillSpan (arXiv 2604.23009)](https://arxiv.org/html/2604.23009v1)
- [Skill Extraction using Weak Supervision (arXiv 2209.08071)](https://arxiv.org/html/2209.08071)
- [Skill-LLM (arXiv 2410.12052)](https://www.alphaxiv.org/abs/2410.12052)

**Fairness / Responsible AI**
- [PopResume (arXiv 2603.22714)](https://arxiv.org/pdf/2603.22714)
- [Gender, Race, and Intersectional Bias in Resume Screening via LM Retrieval (arXiv 2407.20371)](https://arxiv.org/abs/2407.20371)
- [Can LLMs Hire Fairly? Racial Bias in Resume Screening (arXiv 2606.28978)](https://arxiv.org/html/2606.28978)
- [Bias in the Tails (arXiv 2604.19984)](https://arxiv.org/abs/2604.19984)
- [The Audit Decides the Verdict (arXiv 2609.09048)](https://arxiv.org/pdf/2609.09048)
- [Measuring Validity in LLM-based Resume Screening (arXiv 2602.18550)](https://arxiv.org/pdf/2602.18550)

**Portfolio / โค้ดจริง**
- [maxprogerit/resumeScreen](https://github.com/maxprogerit/resumeScreen)
- [dileepreddy27/careermatch-engine](https://github.com/dileepreddy27/careermatch-engine)
- [krtk06/ResumeAnalysis](https://github.com/krtk06/ResumeAnalysis)
- [rahulmurarisetty/resume-evidence-matcher](https://github.com/rahulmurarisetty/resume-evidence-matcher)
- [Sharawey74/Recruiter-Pro](https://github.com/Sharawey74/Recruiter-Pro)
- [clawnify/OpenRecruit](https://github.com/clawnify/OpenRecruit)
- [erAnjaliGoyal/Resume-Screener-LLM](https://github.com/erAnjaliGoyal/Resume-Screener-LLM)
- [adiagrawaal/AI-Resume-Screening-System](https://github.com/adiagrawaal/AI-Resume-Screening-System)
- [suraj07nangare/ai-resume-screening-system](https://github.com/suraj07nangare/ai-resume-screening-system)
- [GitHub topic: resume-matcher](https://github.com/topics/resume-matcher)

**วิธีวัดผล / IAA**
- [RCP-nDCG@10 (Cohere)](https://cohere.com/blog/rcp-ndcg)
- [AI Security Beyond Core Domains — resume screening case study (arXiv 2512.20164)](https://arxiv.org/pdf/2512.20164)
