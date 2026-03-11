# MUSHRA Listening Test - Voice Perception Study

MUSHRA listening test that collects ratings and saves results to Google Sheets.

## File Structure

```
mushra-listening-test/
├── index.html      - Entire test + config (edit the JSON block inside)
├── write.php       - PHP backend: receives results and writes to CSV
├── /audio          - Folder for storing wav-files to use
└── README.md
```

---

## 1. Add Your Audio Files

Drop your WAV files into the same folder as `index.html`. Then update the `src` paths in the JSON config block inside `index.html`:

```json
"reference": { "label": "Reference", "src": "ref.wav" },
"stimuli": [
  { "id": "S1",  "label": "Sample A",         "src": "manip1.wav", "isHiddenRef": false },
  { "id": "S2",  "label": "Sample B",         "src": "manip2.wav", "isHiddenRef": false },
  { "id": "S3",  "label": "Sample C",         "src": "manip3.wav", "isHiddenRef": false },
  { "id": "REF", "label": "Hidden Reference", "src": "ref.wav",    "isHiddenRef": true  }
]
```

> One `"isHiddenRef": true` entry is required per MUSHRA page.

---

## 2. Page Types

Open `index.html` and find the `<script id="cfg">` block. The `pages` array controls the test flow. Five page types are supported:

### `consent`: participant information & consent checkbox
```json
{ "type": "consent", "name": "Participant Information & Consent" }
```

### `demographics`: age, education, native English
```json
{ "type": "demographics", "name": "About You" }
```

### `generic`: text / introduction page
```json
{ "type": "generic", "name": "Introduction", "content": "HTML content here." }
```

### `mushra`: rating page
```json
{
  "type":      "mushra",
  "name":      "Human-likeness",
  "id":        "human_likeness",
  "content":   "Rate how human-like each voice sounds.",
  "randomize": true,
  "reference": { "label": "Reference", "src": "ref.wav" },
  "stimuli": [ ... ]
}
```

### `finish`: thank-you page
```json
{ "type": "finish", "name": "Thank You", "content": "Your responses have been recorded.", "showResults": false }
```

---

## 3. Results

### Google Sheets (automatic)
Set `"remoteService"` in the config to your Google Apps Script web app URL. Results are posted automatically when a participant reaches the finish page.

**CSV / Sheets columns:**
| Column | Description |
|--------|-------------|
| session_test_id | Test identifier from config |
| session_uuid | Unique per participant |
| dem_age | Age group from demographics page |
| dem_education | Education level from demographics page |
| dem_native_english | Native English speaker (Yes/No) |
| trial_id | MUSHRA page identifier |
| rating_stimulus | Stimulus ID (S1, S2, S3, REF) |
| rating_condition | Condition label (low, med, high, reference) |
| rating_score | 0–100 |
| rating_time | Time spent on page (ms) |
| rating_comment | Reserved for future use |

### CSV download (always available)
Participants can download their results as a CSV directly from the finish page, no server needed.

---

## 4. Hosting

Deployed on **GitHub Pages** (HTTPS required for Web Audio API volume boost).

### Local testing
```bash
python3 -m http.server 8080
# open http://localhost:8080
```

---

## 5. Config Options

| Key | Description |
|-----|-------------|
| `testname` | Displayed in the top bar |
| `testId` | Used as identifier in results |
| `showButtonPreviousPage` | Show/hide back button (`true`/`false`) |
| `remoteService` | Google Apps Script URL for Sheets saving |
