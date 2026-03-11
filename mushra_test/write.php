<?php
/**
 * MUSHRA Results Writer — service/write.php
 *
 * Receives a JSON POST from the frontend and appends ratings to a per-test CSV.
 * Also stores the raw JSON for debugging.
 *
 * Folder structure expected:
 *   service/write.php   ← this file
 *   results/            ← created automatically, must be writable
 */

header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

// Preflight
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') { http_response_code(200); exit; }

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['error' => 'Method Not Allowed']);
    exit;
}

// ── Parse body ──────────────────────────────────────────────────────────────
$raw  = file_get_contents('php://input');
$data = json_decode($raw, true);

if (!$data || !isset($data['testId'], $data['sessionId'], $data['pages'])) {
    http_response_code(400);
    echo json_encode(['error' => 'Invalid or incomplete JSON payload']);
    exit;
}

// ── Prepare results directory ────────────────────────────────────────────────
$resultsDir = __DIR__ . '/../results';
if (!is_dir($resultsDir) && !mkdir($resultsDir, 0755, true)) {
    http_response_code(500);
    echo json_encode(['error' => 'Cannot create results directory']);
    exit;
}

$testId    = preg_replace('/[^a-zA-Z0-9_\-]/', '_', $data['testId']);
$csvFile   = $resultsDir . '/' . $testId . '_results.csv';
$jsonFile  = $resultsDir . '/' . $testId . '_raw.jsonl';

// ── Build CSV rows ───────────────────────────────────────────────────────────
$csvHeader = ['session_id','timestamp','test_id','page_id','page_name',
              'position','stimulus_id','is_hidden_ref','score','plays'];

$rows = [];
foreach ($data['pages'] as $page) {
    $pageId   = $page['id']   ?? '';
    $pageName = $page['name'] ?? '';
    foreach ($page['ratings'] as $r) {
        $rows[] = [
            $data['sessionId'],
            $data['timestamp'],
            $data['testId'],
            $pageId,
            $pageName,
            $r['position']    ?? '',
            $r['stimulusId']  ?? '',
            isset($r['isHiddenRef']) && $r['isHiddenRef'] ? 'true' : 'false',
            $r['score'] !== null && $r['score'] !== '' ? $r['score'] : '',
            $r['plays']       ?? 0,
        ];
    }
}

// ── Write CSV (header only if file is new) ───────────────────────────────────
$isNew = !file_exists($csvFile);
$fh    = fopen($csvFile, 'a');
if (!$fh) {
    http_response_code(500);
    echo json_encode(['error' => 'Cannot open CSV file for writing']);
    exit;
}

if ($isNew) {
    fputcsv($fh, $csvHeader);
}
foreach ($rows as $row) {
    fputcsv($fh, $row);
}
fclose($fh);

// ── Append raw JSON line ─────────────────────────────────────────────────────
file_put_contents($jsonFile, json_encode($data) . "\n", FILE_APPEND | LOCK_EX);

// ── Success response ─────────────────────────────────────────────────────────
echo json_encode([
    'ok'      => true,
    'file'    => 'results/' . $testId . '_results.csv',
    'session' => $data['sessionId'],
    'rows'    => count($rows),
]);
