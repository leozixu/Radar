"use strict";

(() => {
    const fields = ["Pt_Azi", "Pt_El", "Pt_Distance", "Pt_Circle", "Batch_ID"];
    const stride = fields.length;
    const emptyPoints = new Float32Array(0);
    const radiansPerDegree = Math.PI / 180;

    const byId = (id) => document.getElementById(id);
    const fileInput = byId("file-input");
    const filePicker = byId("file-picker");
    const dropZone = byId("drop-zone");
    const status = byId("load-status");
    const plotTitle = byId("plot-title");
    const plotStats = byId("plot-stats");
    const emptyState = byId("empty-state");
    const startInput = byId("circle-start");
    const endInput = byId("circle-end");
    const resetRange = byId("reset-range");
    const resetView = byId("reset-view");
    const topView = byId("top-view");
    const calibrationForm = byId("calibration-form");
    const calibrationStatus = byId("calibration-status");
    const coordinateInfo = byId("coordinate-info");
    const showTrajectories = byId("show-trajectories");
    const trackPicker = byId("track-picker");
    const trackStatus = byId("track-status");
    const kinds = ["clutter", "target", "unknown"];
    const toggles = kinds.map((kind) => byId(`show-${kind}`));
    const counters = kinds.map((kind) => byId(`count-${kind}`));
    const files = [];
    const numberFormat = new Intl.NumberFormat("zh-CN");

    let renderer;
    let calibration;
    let activeData = null;
    let worldPoints = emptyPoints;
    let plottedPoints = emptyPoints;
    let selectedTrack = null;
    let loadVersion = 0;

    function report(message, kind = "info") {
        status.textContent = message;
        status.dataset.kind = kind;
    }

    function showPlotMessage(message) {
        emptyState.hidden = !message;
        if (message) {
            emptyState.firstElementChild.textContent = message;
        }
    }

    function unquote(field) {
        const value = (field ?? "").trim();
        return value.startsWith('"') && value.endsWith('"') ? value.slice(1, -1).trim() : value;
    }

    function parseCsv(text) {
        const lines = text.split(/\r\n|\n|\r/);
        const headerIndex = lines.findIndex((line) => line.trim() !== "");
        if (headerIndex < 0) {
            throw new Error("CSV 文件为空。");
        }
        const headers = lines[headerIndex].split(",").map(unquote);
        const indices = fields.map((field) => headers.indexOf(field));
        const timeIndex = headers.indexOf("Pt_RadarTime");
        const missing = fields.filter((_, index) => indices[index] < 0);
        if (missing.length) {
            throw new Error(`CSV 缺少字段 ${missing.join("、")}。`);
        }

        const points = new Float32Array((lines.length - headerIndex - 1) * stride);
        const tracks = new Map();
        let offset = 0;
        let targetCount = 0;
        let minCircle = Infinity;
        let maxCircle = -Infinity;
        for (let lineIndex = headerIndex + 1; lineIndex < lines.length; lineIndex += 1) {
            const line = lines[lineIndex];
            if (!line.trim()) {
                continue;
            }
            const row = line.split(",");
            for (let index = 0; index < stride; index += 1) {
                const value = unquote(row[indices[index]]);
                const numeric = Number(value);
                if (!value || !Number.isFinite(numeric)) {
                    throw new Error(`第 ${lineIndex + 1} 行的 ${fields[index]} 不是有效数字。`);
                }
                points[offset + index] = numeric;
                if (index === 3) {
                    if (!Number.isSafeInteger(numeric) || numeric < 0 || points[offset + index] !== numeric) {
                        throw new Error(`第 ${lineIndex + 1} 行的 Pt_Circle 超出支持的圈号范围。`);
                    }
                    minCircle = Math.min(minCircle, numeric);
                    maxCircle = Math.max(maxCircle, numeric);
                }
                if (index === 4) {
                    if (!Number.isSafeInteger(numeric) || numeric < -1) {
                        throw new Error(`第 ${lineIndex + 1} 行的 Batch_ID 无效。`);
                    }
                    points[offset + index] = numeric < 0 ? 2 : Number(numeric > 0);
                    if (numeric > 0) {
                        let observations = tracks.get(numeric);
                        if (!observations) {
                            observations = [];
                            tracks.set(numeric, observations);
                        }
                        const time = Number(unquote(row[timeIndex]));
                        observations.push({ offset, time: Number.isFinite(time) ? time : 0 });
                        targetCount += 1;
                    }
                }
            }
            offset += stride;
        }
        if (!offset) {
            throw new Error("CSV 只有表头，缺少点迹记录。");
        }
        for (const observations of tracks.values()) {
            observations.sort((a, b) => points[a.offset + 3] - points[b.offset + 3] || a.time - b.time);
        }
        return { points: points.subarray(0, offset), minCircle, maxCircle, tracks, targetCount };
    }

    function readCalibration() {
        const numberFrom = (id) => byId(id).valueAsNumber;
        const azimuthMin = numberFrom("azimuth-min");
        const azimuthMax = numberFrom("azimuth-max");
        const elevationMin = numberFrom("elevation-min");
        const elevationMax = numberFrom("elevation-max");
        const rangeMinKm = numberFrom("range-min-km");
        const rangeMaxInput = byId("range-max-km");
        const rangeMaxKm = rangeMaxInput.value === "" && !rangeMaxInput.validity.badInput
            ? null : rangeMaxInput.valueAsNumber;
        const azimuthSpan = azimuthMax - azimuthMin;
        if (!(azimuthSpan > 0 && azimuthSpan <= 360)) {
            throw new Error("方位角上限应大于下限，且跨度不能超过 360°。");
        }
        if (!(elevationMin >= -90 && elevationMax <= 90 && elevationMax > elevationMin)) {
            throw new Error("俯仰角范围应在 −90° 至 90° 内，且上限应大于下限。");
        }
        if (!(rangeMinKm >= 0 && Number.isFinite(rangeMinKm)) ||
            (rangeMaxKm === null ? rangeMinKm !== 0 : !(rangeMaxKm > rangeMinKm && Number.isFinite(rangeMaxKm)))) {
            throw new Error("距离应为非负数，且上限应大于下限。量程未知时请将下限设为 0，并留空上限。");
        }
        return {
            azimuthStart: (azimuthMin % 360) * radiansPerDegree,
            azimuthSpan: azimuthSpan * radiansPerDegree,
            elevationStart: elevationMin * radiansPerDegree,
            elevationSpan: (elevationMax - elevationMin) * radiansPerDegree,
            minHeight: Math.sin(Math.min(0, elevationMin) * radiansPerDegree),
            maxHeight: Math.sin(Math.max(0, elevationMax) * radiansPerDegree),
            minRadius: rangeMaxKm === null ? 0 : rangeMinKm / rangeMaxKm,
            maxRangeKm: rangeMaxKm,
        };
    }

    function toWorldPoints(points) {
        const { azimuthStart, azimuthSpan, elevationStart, elevationSpan, minRadius } = calibration;
        const world = new Float32Array(points.length);
        const radiusSpan = 1 - minRadius;
        for (let offset = 0; offset < points.length; offset += stride) {
            const azimuth = azimuthStart + points[offset] * azimuthSpan;
            const elevation = elevationStart + points[offset + 1] * elevationSpan;
            const radius = minRadius + points[offset + 2] * radiusSpan;
            const horizontal = radius * Math.cos(elevation);
            // The renderer stores vertical Z between the two horizontal axes X and Y.
            world[offset] = horizontal * Math.sin(azimuth);
            world[offset + 1] = radius * Math.sin(elevation);
            world[offset + 2] = horizontal * Math.cos(azimuth);
            world[offset + 3] = points[offset + 3];
            world[offset + 4] = points[offset + 4];
        }
        return world;
    }

    function applyCalibration(next) {
        calibration = next;
        renderer.setWorld({ maxRangeKm: next.maxRangeKm, minHeight: next.minHeight, maxHeight: next.maxHeight });
        coordinateInfo.textContent = next.maxRangeKm === null ? "等比例 · 距离归一化" : "等比例 · 单位 km";
        if (activeData) {
            worldPoints = toWorldPoints(activeData.points);
            updateSelectedTrack();
            updateFilter();
        }
    }

    function setRangeEnabled(enabled) {
        startInput.disabled = !enabled;
        endInput.disabled = !enabled;
        resetRange.disabled = !enabled;
    }

    function circleValue(input, fallback) {
        const value = Number.isFinite(input.valueAsNumber) ? Math.trunc(input.valueAsNumber) : fallback;
        return Math.min(activeData.maxCircle, Math.max(activeData.minCircle, value));
    }

    function appendDashedSegment(segments, points, from, to) {
        const x = points[from];
        const y = points[from + 1];
        const z = points[from + 2];
        const dx = points[to] - x;
        const dy = points[to + 1] - y;
        const dz = points[to + 2] - z;
        const dashCount = Math.max(2, Math.min(8, Math.ceil(Math.hypot(dx, dy, dz) / 0.022)));
        const units = dashCount * 2 - 1;
        for (let dash = 0; dash < dashCount; dash += 1) {
            const start = 2 * dash / units;
            const end = (2 * dash + 1) / units;
            segments.push(x + dx * start, y + dy * start, z + dz * start,
                x + dx * end, y + dy * end, z + dz * end);
        }
    }

    function updateTrajectories(start, end) {
        const { tracks } = activeData;
        const points = worldPoints;
        const groups = selectedTrack ? [selectedTrack] : tracks.values();
        const solid = [];
        const dashed = [];
        let trackCount = 0;
        let observationCount = 0;
        const showLines = showTrajectories.checked;

        for (const observations of groups) {
            let previous = null;
            for (const { offset } of observations) {
                const circle = points[offset + 3];
                if (circle < start) {
                    continue;
                }
                if (circle > end) {
                    break;
                }
                if (previous === null) {
                    trackCount += 1;
                } else if (showLines) {
                    if (circle - points[previous + 3] > 1) {
                        appendDashedSegment(dashed, points, previous, offset);
                    } else {
                        solid.push(points[previous], points[previous + 1], points[previous + 2],
                            points[offset], points[offset + 1], points[offset + 2]);
                    }
                }
                previous = offset;
                observationCount += 1;
            }
        }

        renderer.setTrajectories(new Float32Array(solid), new Float32Array(dashed), selectedTrack !== null);
        if (!tracks.size) {
            trackStatus.textContent = "此文件没有标注航迹";
        } else if (selectedTrack) {
            trackStatus.textContent = `范围内 ${numberFormat.format(observationCount)} 个观测点`;
        } else {
            trackStatus.textContent = `范围内 ${numberFormat.format(trackCount)} 条航迹`;
        }
    }

    function updateSelectedTrack() {
        selectedTrack = activeData.tracks.get(Number(trackPicker.value)) ?? null;
        const { targetCount } = activeData;
        const points = worldPoints;
        plottedPoints = points;
        if (selectedTrack) {
            const focusedPoints = new Float32Array(points.length + (selectedTrack.length - targetCount) * stride);
            let nextOffset = 0;
            function copyPoint(offset) {
                for (let field = 0; field < stride; field += 1) {
                    focusedPoints[nextOffset + field] = points[offset + field];
                }
                nextOffset += stride;
            }
            for (let offset = 0; offset < points.length; offset += stride) {
                if (points[offset + 4] !== 1) {
                    copyPoint(offset);
                }
            }
            for (const { offset } of selectedTrack) {
                copyPoint(offset);
            }
            plottedPoints = focusedPoints;
        }
        renderer.setData(plottedPoints);
    }

    function updateFilter(changedInput) {
        if (!activeData) {
            return;
        }
        let start = circleValue(startInput, activeData.minCircle);
        let end = circleValue(endInput, activeData.maxCircle);
        if (start > end) {
            if (changedInput === endInput) {
                start = end;
            } else {
                end = start;
            }
        }
        startInput.value = start;
        endInput.value = end;

        const [showClutter, showTarget, showUnknown] = toggles.map((input) => input.checked);
        renderer.setFilter({ minCircle: start, maxCircle: end, showClutter, showTarget, showUnknown });
        const counts = [0, 0, 0];
        const points = plottedPoints;
        for (let index = 0; index < points.length; index += stride) {
            const circle = points[index + 3];
            if (circle >= start && circle <= end) {
                counts[points[index + 4]] += 1;
            }
        }
        let visible = 0;
        counts.forEach((count, index) => {
            counters[index].textContent = numberFormat.format(count);
            if (toggles[index].checked) {
                visible += count;
            }
        });
        plotStats.textContent = `${numberFormat.format(visible)} / ${numberFormat.format(activeData.points.length / stride)} 点`;
        updateTrajectories(start, end);
    }

    function selectAllCircles() {
        if (!activeData) {
            return;
        }
        startInput.value = activeData.minCircle;
        endInput.value = activeData.maxCircle;
        updateFilter();
    }

    async function loadFile(index) {
        const file = files[index];
        const version = ++loadVersion;
        activeData = null;
        worldPoints = emptyPoints;
        plottedPoints = emptyPoints;
        selectedTrack = null;
        renderer.setData(emptyPoints);
        renderer.setTrajectories(emptyPoints, emptyPoints);
        trackPicker.length = 1;
        trackPicker.value = "";
        trackPicker.disabled = true;
        showTrajectories.disabled = true;
        trackStatus.textContent = "";
        setRangeEnabled(false);
        startInput.value = "";
        endInput.value = "";
        counters.forEach((counter) => { counter.textContent = "—"; });
        plotTitle.textContent = file.name;
        plotStats.textContent = "— 点";
        showPlotMessage("正在读取点迹…");
        report(`正在读取 ${file.name}…`);
        try {
            const content = await file.text();
            if (version !== loadVersion) {
                return;
            }
            activeData = parseCsv(content);
            worldPoints = toWorldPoints(activeData.points);
            const batchIds = Array.from(activeData.tracks.keys()).sort((a, b) => a - b);
            for (const id of batchIds) {
                trackPicker.add(new Option(`批号 ${id}`, String(id)));
            }
            trackPicker.disabled = !batchIds.length;
            showTrajectories.disabled = !batchIds.length;
            updateSelectedTrack();
            for (const input of [startInput, endInput]) {
                input.min = activeData.minCircle;
                input.max = activeData.maxCircle;
            }
            setRangeEnabled(true);
            selectAllCircles();
            showPlotMessage("");
            report(`已加载 ${file.name}`);
        } catch (error) {
            if (version === loadVersion) {
                activeData = null;
                showPlotMessage("无法显示此文件");
                report(`无法加载 ${file.name}。${error.message}`, "error");
            }
        }
    }

    function importFiles(selectedFiles) {
        const csvFiles = Array.from(selectedFiles).filter((file) => /\.csv$/i.test(file.name));
        if (!csvFiles.length) {
            report("请选择 DATA 中的 CSV 文件。", "error");
            return;
        }
        const firstNewIndex = files.length;
        for (const file of csvFiles) {
            const option = new Option(file.name, String(files.length));
            files.push(file);
            filePicker.add(option);
        }
        filePicker.disabled = false;
        filePicker.value = String(firstNewIndex);
        loadFile(firstNewIndex);
    }

    try {
        renderer = window.createRadarCloud(byId("plot-canvas"), byId("axes-canvas"));
        applyCalibration(readCalibration());
    } catch (error) {
        showPlotMessage("三维视图无法启动");
        byId("import-button").disabled = true;
        resetView.disabled = true;
        topView.disabled = true;
        calibrationForm.querySelector("button[type='submit']").disabled = true;
        report(error.message, "error");
        return;
    }

    byId("import-button").addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) {
            importFiles(fileInput.files);
            fileInput.value = "";
        }
    });
    filePicker.addEventListener("change", () => {
        if (filePicker.value !== "") {
            loadFile(Number(filePicker.value));
        }
    });
    dropZone.addEventListener("dragover", (event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
        dropZone.classList.add("is-dragging");
    });
    dropZone.addEventListener("dragleave", (event) => {
        if (!dropZone.contains(event.relatedTarget)) {
            dropZone.classList.remove("is-dragging");
        }
    });
    dropZone.addEventListener("drop", (event) => {
        event.preventDefault();
        dropZone.classList.remove("is-dragging");
        importFiles(event.dataTransfer.files);
    });
    for (const input of [...toggles, startInput, endInput]) {
        input.addEventListener("change", () => updateFilter(input));
    }
    resetRange.addEventListener("click", selectAllCircles);
    resetView.addEventListener("click", () => renderer.resetView());
    topView.addEventListener("click", () => renderer.setTopView());
    calibrationForm.addEventListener("input", () => {
        calibrationStatus.textContent = "";
    });
    calibrationForm.addEventListener("submit", (event) => {
        event.preventDefault();
        try {
            applyCalibration(readCalibration());
            calibrationStatus.textContent = "已应用换算参数。";
            calibrationStatus.dataset.kind = "info";
        } catch (error) {
            calibrationStatus.textContent = error.message;
            calibrationStatus.dataset.kind = "error";
        }
    });
    showTrajectories.addEventListener("change", () => updateFilter());
    trackPicker.addEventListener("change", () => {
        if (activeData) {
            updateSelectedTrack();
            updateFilter();
        }
    });
})();
