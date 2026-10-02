"""Auditable dataset/source manifest helpers; no download or training logic."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
from typing import Iterable


@dataclass(frozen=True)
class FileRecord:
    path: str
    size_bytes: int
    sha256: str


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_files(root: Path, *, suffixes: Iterable[str] | None = None) -> tuple[FileRecord, ...]:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"数据目录不存在：{root}")
    allowed = None if suffixes is None else {str(x).lower() for x in suffixes}
    records = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if allowed is not None and path.suffix.lower() not in allowed:
            continue
        records.append(FileRecord(str(path.relative_to(root)), path.stat().st_size, sha256_file(path)))
    return tuple(records)


def _is_unresolved(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return not text or text in {"unknown", "pending", "待核实", "待下载后计算"}


def audit_manifest(manifest: dict, *, project_root: Path, verify_archives: bool = False) -> dict:
    """Audit provenance and enforce the pre-training data gate.

    ``ok`` means no blocking provenance/download error exists.  A source being
    publicly described is not enough: a primary training source must also have
    a verified licence, pinned version, local raw directory, download date and
    checksum before ``ready_for_training`` can become true.
    """
    required = ("name", "status", "source_url", "license", "version")
    findings = []
    primary_sources = []
    for source in manifest.get("sources", []):
        name = source.get("name", "未命名")
        missing = [key for key in required if _is_unresolved(source.get(key))]
        if missing:
            findings.append({"source": name, "severity": "review", "missing": missing})
        if _is_unresolved(source.get("license")) or not source.get("license_verified", False):
            findings.append({"source": name, "severity": "review", "message": "许可证尚未核实"})

        if source.get("role") != "primary":
            archive_path_value = source.get("archive_local_path")
            if archive_path_value:
                archive_path = project_root / archive_path_value
                archive_findings = []
                if not archive_path.is_file():
                    archive_findings.append("已登记的候选原始归档不存在")
                else:
                    expected_bytes = source.get("archive_bytes")
                    if expected_bytes is not None and archive_path.stat().st_size != expected_bytes:
                        archive_findings.append("候选原始归档大小与登记不一致")
                    if verify_archives and source.get("archive_sha256"):
                        if sha256_file(archive_path) != source["archive_sha256"]:
                            archive_findings.append("候选原始归档 SHA-256 与官方值不一致")
                if archive_findings:
                    findings.extend(
                        {"source": name, "severity": "review", "message": message}
                        for message in archive_findings
                    )
                else:
                    findings.append({
                        "source": name,
                        "severity": "info",
                        "message": "候选原始归档存在，大小与登记一致"
                        + ("，SHA-256 已复核" if verify_archives and source.get("archive_sha256") else ""),
                    })
            if source.get("training_eligible") is False and source.get("training_blocker"):
                findings.append({
                    "source": name,
                    "severity": "info",
                    "message": source["training_blocker"],
                })
            continue
        primary_sources.append(name)
        if _is_unresolved(source.get("download_date")):
            findings.append({"source": name, "severity": "error", "message": "主数据集尚未下载"})
        if _is_unresolved(source.get("checksum")):
            findings.append({"source": name, "severity": "error", "message": "主数据集尚未固定校验值"})
        archive_path_value = source.get("archive_local_path")
        if archive_path_value:
            archive_path = project_root / archive_path_value
            if not archive_path.is_file():
                findings.append({"source": name, "severity": "error", "message": "已登记的原始归档不存在"})
            else:
                expected_bytes = source.get("archive_bytes")
                if expected_bytes is not None and archive_path.stat().st_size != expected_bytes:
                    findings.append({"source": name, "severity": "error", "message": "原始归档大小与登记不一致"})
                if verify_archives and source.get("archive_sha256"):
                    if sha256_file(archive_path) != source["archive_sha256"]:
                        findings.append({"source": name, "severity": "error", "message": "原始归档 SHA-256 与官方值不一致"})
        local_root = source.get("local_root")
        if not local_root or not (project_root / local_root).is_dir():
            findings.append({"source": name, "severity": "error", "message": "主数据集本地原始目录不存在"})
        training_root = source.get("training_data_root")
        if not training_root or not (project_root / training_root).is_dir():
            findings.append({"source": name, "severity": "error", "message": "主数据集训练数据目录不存在"})
        verification_path = source.get("training_verification_path")
        if not verification_path or not (project_root / verification_path).is_file():
            findings.append({"source": name, "severity": "error", "message": "主数据集缺少训练数据验证报告"})
        else:
            try:
                verification = json.loads(
                    (project_root / verification_path).read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                findings.append({
                    "source": name,
                    "severity": "error",
                    "message": f"训练数据验证报告无法读取：{exc}",
                })
            else:
                if not verification.get("ok") or verification.get("findings"):
                    findings.append({"source": name, "severity": "error", "message": "训练数据验证报告未通过"})
                if verification.get("dataset") != name or str(verification.get("dataset_version")) != str(source.get("version")):
                    findings.append({"source": name, "severity": "error", "message": "训练数据验证报告的数据集或版本不匹配"})
                if not verification.get("verified_recording_files") or not verification.get("verified_window_rows"):
                    findings.append({"source": name, "severity": "error", "message": "训练数据验证报告没有有效样本"})

    if not primary_sources:
        findings.append({"source": "数据清单", "severity": "error", "message": "尚未选定主数据集"})
    local = manifest.get("local_inventory", {})
    if local:
        root = project_root / local.get("root", "")
        records = inventory_files(root, suffixes=local.get("suffixes"))
        expected = local.get("files", [])
        findings.append({"source": local.get("name", "本地数据"), "severity": "info", "file_count": len(records), "expected_record_count": len(expected)})
        actual_by_path = {record.path: record for record in records}
        expected_by_path = {record.get("path"): record for record in expected}
        if set(actual_by_path) != set(expected_by_path):
            findings.append({"source": local.get("name", "本地数据"), "severity": "error", "message": "本地文件清单与登记不一致"})
        for path, expected_record in expected_by_path.items():
            actual = actual_by_path.get(path)
            if actual is None:
                continue
            if actual.size_bytes != expected_record.get("size_bytes") or actual.sha256 != expected_record.get("sha256"):
                findings.append({"source": local.get("name", "本地数据"), "severity": "error", "path": path, "message": "本地文件大小或 SHA-256 与登记不一致"})

    ready = bool(primary_sources) and not any(item["severity"] == "error" for item in findings)
    return {
        "ok": ready,
        "ready_for_training": ready,
        "primary_sources": primary_sources,
        "findings": findings,
    }


def write_audit_report(manifest_path: Path, output_path: Path, *, project_root: Path) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    report = audit_manifest(manifest, project_root=project_root, verify_archives=True)
    report["manifest"] = str(Path(manifest_path))
    Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


__all__ = ["FileRecord", "audit_manifest", "inventory_files", "sha256_file", "write_audit_report"]
