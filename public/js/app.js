/**
 * AMR-Predict Application Logic v2.0
 * Clinical AI Intelligence Workspace & Database Persistence
 *
 * REST Endpoints:
 *  GET  /api/config        — system configuration & clinical presets
 *  POST /api/predict       — multi-model resistance prediction + auto-save to DB
 *  POST /api/explain       — SHAP waterfall + factor attributions
 *  GET  /api/history       — historical prediction records (search/filter)
 *  GET  /api/history/{id}  — full historical record details
 *  GET  /api/patients      — patient directory
 *  GET  /api/reports       — clinical antibiogram reports
 *  GET  /api/reports/{id}  — full report details for preview/export
 *  GET  /api/stats         — live database stats
 *  GET  /api/models        — model registry & benchmark metrics
 *  GET  /api/eda           — dataset distribution insights
 */

// ─── Global State ───────────────────────────────────────────────────────────
const state = {
    config: null,
    currentPatient: null,
    currentPredictions: null,
    selectedAntibiotic: "Ciprofloxacin",
    shapMode: "local",
    modelsData: null,
    edaData: null,
    chartInstances: {},
    currentPage: "page-overview",
};

// ─── AMR Color Palette ──────────────────────────────────────────────────────
const AMR_PALETTE = [
    "#3D52A0",
    "#7091E6",
    "#8697C4",
    "#ADBBDA",
    "#5A73C2",
    "#7E98D6",
    "#4B62AF",
    "#9BAAD1",
];

const AMR_PALETTE_ALPHA = (hex, a) => {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${a})`;
};

// ─── Application Initialization ─────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
    setupSidebarNavigation();
    setupMobileNavigation();
    setGreeting();

    // Wire up Dashboard CTA buttons
    ["btnNewAnalysis", "btnDashAnalyze", "btnHeroCta"].forEach(id => {
        document.getElementById(id)?.addEventListener("click", () => navigateTo("page-analyzer"));
    });

    document.getElementById("btnDashModel")?.addEventListener("click", () => navigateTo("page-model"));

    // Wire up buttons with data-page navigation
    document.querySelectorAll("button[data-page]").forEach(btn => {
        btn.addEventListener("click", () => navigateTo(btn.dataset.page));
    });

    // Form and control handlers
    setupFormListeners();
    setupFactorCards();
    setupExplainListeners();
    setupHistoryAndReportsListeners();
    setupModalListeners();

    // Load initial data
    await loadSystemConfig();
    await loadDatabaseStats();
    await loadRecentAnalyses();
    await loadPredictionHistory();
    await loadPatientsDirectory();
    await loadClinicalReports();
    await loadModelBenchmarks();
    await loadDatasetEda();

    // Trigger initial prediction with default form data
    const form = document.getElementById("patientForm");
    if (form) {
        form.dispatchEvent(new Event("submit"));
    }
});

// ─── Dynamic Greeting ────────────────────────────────────────────────────────
function setGreeting() {
    const hour = new Date().getHours();
    let greeting = "Good morning";
    if (hour >= 12 && hour < 17) greeting = "Good afternoon";
    else if (hour >= 17) greeting = "Good evening";
    const el = document.getElementById("dashGreeting");
    if (el) el.textContent = `${greeting}, Clinician`;
}

// ─── Navigation & View Routing ───────────────────────────────────────────────
function navigateTo(pageId) {
    document.querySelectorAll(".page-view").forEach(p => {
        p.classList.remove("active");
        p.setAttribute("aria-hidden", "true");
    });

    const target = document.getElementById(pageId);
    if (target) {
        target.classList.add("active");
        target.removeAttribute("aria-hidden");
    }

    document.querySelectorAll(".nav-item").forEach(item => {
        const isActive = item.dataset.page === pageId;
        item.classList.toggle("active", isActive);
        if (isActive) item.setAttribute("aria-current", "page");
        else item.removeAttribute("aria-current");
    });

    const navItem = document.querySelector(`.nav-item[data-page="${pageId}"]`);
    const topbarTitle = document.getElementById("topbarPageTitle");
    if (navItem && topbarTitle) {
        topbarTitle.textContent = navItem.textContent.trim().split("\n")[0].trim();
    }

    state.currentPage = pageId;

    // Refresh view-specific dynamic data
    if (pageId === "page-history") {
        loadPredictionHistory();
    } else if (pageId === "page-reports") {
        loadClinicalReports();
    } else if (pageId === "page-patients") {
        loadPatientsDirectory();
    } else if (pageId === "page-overview") {
        loadDatabaseStats();
        loadRecentAnalyses();
    } else if (pageId === "page-analytics") {
        resizeCharts(["edaResistanceChart", "edaOrganismChart", "edaInfectionChart", "edaComorbiditiesChart"]);
    } else if (pageId === "page-model") {
        resizeCharts(["rocChartCanvas", "calibChartCanvas"]);
    }

    closeMobileSidebar();
}

function setupSidebarNavigation() {
    document.querySelectorAll(".nav-item[data-page]").forEach(item => {
        item.addEventListener("click", () => navigateTo(item.dataset.page));
    });
}

function setupMobileNavigation() {
    const toggle = document.getElementById("sidebarToggle");
    const sidebar = document.getElementById("sidebar");
    const overlay = document.getElementById("sidebarOverlay");

    toggle?.addEventListener("click", () => {
        sidebar?.classList.toggle("open");
        overlay?.classList.toggle("visible");
        overlay?.setAttribute("aria-hidden", !overlay.classList.contains("visible"));
    });

    overlay?.addEventListener("click", closeMobileSidebar);
}

function closeMobileSidebar() {
    document.getElementById("sidebar")?.classList.remove("open");
    const overlay = document.getElementById("sidebarOverlay");
    overlay?.classList.remove("visible");
    overlay?.setAttribute("aria-hidden", "true");
}

// ─── Clinical Risk Factor Checkbox Toggle Cards ──────────────────────────────
function setupFactorCards() {
    document.querySelectorAll(".factor-card").forEach(card => {
        const checkbox = card.querySelector("input[type='checkbox']");
        if (!checkbox) return;

        updateFactorCardState(card, checkbox.checked);

        checkbox.addEventListener("change", () => {
            updateFactorCardState(card, checkbox.checked);
        });
    });
}

function updateFactorCardState(card, checked) {
    card.classList.toggle("checked", checked);
}

// ─── Form & Preset Listeners ─────────────────────────────────────────────────
function setupFormListeners() {
    const form = document.getElementById("patientForm");
    const presetSelect = document.getElementById("presetSelect");

    if (presetSelect) {
        presetSelect.addEventListener("change", (e) => {
            const presetId = e.target.value;
            if (!presetId || !state.config?.presets) return;

            const preset = state.config.presets.find(p => p.id === presetId);
            if (preset?.data) {
                populateFormWithData(preset.data);
                form?.dispatchEvent(new Event("submit"));
            }
        });
    }

    if (form) {
        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            await runPatientAnalysis();
        });
    }
}

// ─── SHAP Explainability Listeners ───────────────────────────────────────────
function setupExplainListeners() {
    const abxSelect = document.getElementById("explainAbxSelect");
    const btnLocal  = document.getElementById("btnLocalShap");
    const btnGlobal = document.getElementById("btnGlobalShap");

    abxSelect?.addEventListener("change", (e) => {
        state.selectedAntibiotic = e.target.value;
        updateShapExplanationView();
    });

    btnLocal?.addEventListener("click", () => {
        btnLocal.classList.add("active");
        btnLocal.setAttribute("aria-pressed", "true");
        btnGlobal?.classList.remove("active");
        btnGlobal?.setAttribute("aria-pressed", "false");
        state.shapMode = "local";
        document.getElementById("shapChartTitle").textContent =
            `SHAP Local Waterfall — ${state.selectedAntibiotic}`;
        updateShapExplanationView();
    });

    btnGlobal?.addEventListener("click", () => {
        btnGlobal.classList.add("active");
        btnGlobal.setAttribute("aria-pressed", "true");
        btnLocal?.classList.remove("active");
        btnLocal?.setAttribute("aria-pressed", "false");
        state.shapMode = "global";
        document.getElementById("shapChartTitle").textContent =
            `Dataset Global Feature Importance — ${state.selectedAntibiotic}`;
        renderGlobalShapChart();
    });
}

// ─── History & Reports Filter Listeners ───────────────────────────────────────
function setupHistoryAndReportsListeners() {
    // Prediction History search & filter
    const historySearch = document.getElementById("historySearchInput");
    const historyRisk   = document.getElementById("historyRiskFilter");
    const btnRefresh    = document.getElementById("btnRefreshHistory");

    const triggerHistoryReload = () => {
        loadPredictionHistory(historySearch?.value.trim(), historyRisk?.value);
    };

    historySearch?.addEventListener("input", debounce(triggerHistoryReload, 250));
    historyRisk?.addEventListener("change", triggerHistoryReload);
    btnRefresh?.addEventListener("click", triggerHistoryReload);

    // Patient search
    const patientSearch = document.getElementById("patientSearchInput");
    patientSearch?.addEventListener("input", debounce(() => {
        const term = patientSearch.value.toLowerCase().trim();
        document.querySelectorAll("#patientsTableBody tr").forEach(row => {
            const text = row.textContent.toLowerCase();
            row.style.display = text.includes(term) ? "" : "none";
        });
    }, 200));

    // Reports search
    const reportSearch = document.getElementById("reportSearchInput");
    reportSearch?.addEventListener("input", debounce(() => {
        loadClinicalReports(reportSearch.value.trim());
    }, 250));

    // Global Topbar Search
    const globalSearch = document.getElementById("globalSearchInput");
    globalSearch?.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            const q = globalSearch.value.trim();
            if (q) {
                navigateTo("page-history");
                const hi = document.getElementById("historySearchInput");
                if (hi) { hi.value = q; loadPredictionHistory(q); }
            }
        }
    });
}

// ─── Modal Listeners (Report Preview) ─────────────────────────────────────────
function setupModalListeners() {
    const modal = document.getElementById("reportModal");
    const btnClose = document.getElementById("btnCloseReportModal");
    const btnDismiss = document.getElementById("btnDismissReport");
    const btnPrint = document.getElementById("btnPrintReport");

    const closeModal = () => modal?.classList.remove("open");

    btnClose?.addEventListener("click", closeModal);
    btnDismiss?.addEventListener("click", closeModal);
    modal?.addEventListener("click", (e) => {
        if (e.target === modal) closeModal();
    });

    btnPrint?.addEventListener("click", () => {
        window.print();
    });
}

// ─── Load System Config ───────────────────────────────────────────────────────
async function loadSystemConfig() {
    try {
        const res = await fetch("/api/config");
        if (!res.ok) throw new Error("Failed to load config");
        state.config = await res.json();

        const presetSelect = document.getElementById("presetSelect");
        if (presetSelect && state.config.presets) {
            presetSelect.innerHTML = `<option value="">— Preset —</option>`;
            state.config.presets.forEach(p => {
                const opt = document.createElement("option");
                opt.value = p.id;
                opt.textContent = p.name;
                presetSelect.appendChild(opt);
            });
        }
    } catch (err) {
        console.error("Config load error:", err);
    }
}

// ─── Populate Form with Data ─────────────────────────────────────────────────
function populateFormWithData(data) {
    document.getElementById("inputPatientId").value   = data.patient_id || "";
    document.getElementById("inputAge").value         = data.age;
    document.getElementById("inputSex").value         = data.sex;
    document.getElementById("inputCreatinine").value  = data.creatinine_mg_dl;
    document.getElementById("inputWbc").value         = data.wbc_count_k_ul;
    document.getElementById("inputInfection").value   = data.infection_type;
    document.getElementById("inputOrganism").value    = data.organism;
    document.getElementById("inputPriorUti").value    = data.num_prior_uti_1yr;

    const flags = {
        inputHospital:         "recent_hospitalization_90d",
        inputAbx:              "recent_antibiotic_use_90d",
        inputResistantCulture: "prior_resistant_culture_1yr",
        inputDiabetes:         "diabetes",
        inputCatheter:         "catheter_use",
        inputImmuno:           "immunocompromised",
        inputNursing:          "nursing_home_resident",
        inputTravel:           "travel_last_6mo",
        inputHcw:              "healthcare_worker",
    };

    Object.entries(flags).forEach(([id, key]) => {
        const el = document.getElementById(id);
        if (el) {
            el.checked = Boolean(data[key]);
            const card = el.closest(".factor-card");
            if (card) updateFactorCardState(card, el.checked);
        }
    });
}

// ─── Collect Form Data ────────────────────────────────────────────────────────
function getFormData() {
    return {
        patient_id:                document.getElementById("inputPatientId").value || undefined,
        age:                        parseFloat(document.getElementById("inputAge").value) || 50,
        sex:                        document.getElementById("inputSex").value,
        infection_type:             document.getElementById("inputInfection").value,
        organism:                   document.getElementById("inputOrganism").value,
        diabetes:                   document.getElementById("inputDiabetes").checked ? 1 : 0,
        recent_hospitalization_90d: document.getElementById("inputHospital").checked ? 1 : 0,
        recent_antibiotic_use_90d:  document.getElementById("inputAbx").checked ? 1 : 0,
        num_prior_uti_1yr:          parseInt(document.getElementById("inputPriorUti").value) || 0,
        catheter_use:               document.getElementById("inputCatheter").checked ? 1 : 0,
        immunocompromised:          document.getElementById("inputImmuno").checked ? 1 : 0,
        nursing_home_resident:      document.getElementById("inputNursing").checked ? 1 : 0,
        prior_resistant_culture_1yr:document.getElementById("inputResistantCulture").checked ? 1 : 0,
        creatinine_mg_dl:           parseFloat(document.getElementById("inputCreatinine").value) || 1.0,
        wbc_count_k_ul:             parseFloat(document.getElementById("inputWbc").value) || 9.5,
        travel_last_6mo:            document.getElementById("inputTravel").checked ? 1 : 0,
        healthcare_worker:          document.getElementById("inputHcw").checked ? 1 : 0,
    };
}

// ─── Main Inference Action: Predict & Save to DB ─────────────────────────────
async function runPatientAnalysis() {
    const btn         = document.getElementById("btnAnalyze");
    const latencyText = document.getElementById("latencyText");
    const grid        = document.getElementById("resultsGrid");

    try {
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" style="animation:spin 1s linear infinite;">
                    <path d="M21 12a9 9 0 11-18 0 9 9 0 0118 0"/>
                </svg>
                Evaluating 8 Models…`;
        }
        if (latencyText) latencyText.textContent = "Evaluating models & archiving to DB…";

        const patientData = getFormData();
        state.currentPatient = patientData;

        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(patientData),
        });

        if (!res.ok) {
            const errJson = await res.json();
            throw new Error(errJson.detail || "Prediction request failed");
        }

        const data = await res.json();
        state.currentPredictions = data.predictions;

        if (latencyText) {
            latencyText.textContent = `Saved (${data.prediction_id || "DB"}) · ${data.execution_time_display || data.execution_time_ms + " ms"}`;
        }

        renderAntibioticCards(data.predictions);

        const shapPanel = document.getElementById("shapPanel");
        if (shapPanel) shapPanel.style.display = "block";

        await updateShapExplanationView();

        // Refresh stats & recent list in background
        loadDatabaseStats();
        loadRecentAnalyses();

    } catch (err) {
        console.error("Inference Error:", err);
        if (latencyText) latencyText.textContent = "Analysis error";
        if (grid) {
            grid.innerHTML = `
                <div class="empty-state" style="grid-column:1/-1; padding:36px 16px;">
                    <div class="empty-icon" aria-hidden="true">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
                            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                        </svg>
                    </div>
                    <div class="empty-title">Analysis Error</div>
                    <div class="empty-desc">${err.message}</div>
                </div>`;
        }
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round">
                    <path d="M12 2l2 7h7l-5.5 4 2 7L12 16l-5.5 4 2-7L3 9h7z"/>
                </svg>
                Analyze Resistance Risk`;
        }
    }
}

// ─── Render 8 Antibiotic Result Cards ────────────────────────────────────────
function renderAntibioticCards(predictions) {
    const grid = document.getElementById("resultsGrid");
    if (!grid) return;

    if (!predictions?.length) {
        grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1;"><p>No predictions returned.</p></div>`;
        return;
    }

    grid.innerHTML = "";

    predictions.forEach(p => {
        const riskClass = p.risk_category.toLowerCase();
        const pct       = (p.estimated_resistance_probability * 100).toFixed(1);

        let badgeClass = "risk-badge-low";
        if (riskClass === "high")     badgeClass = "risk-badge-high";
        if (riskClass === "moderate") badgeClass = "risk-badge-moderate";

        let fillClass = "risk-fill-low";
        if (riskClass === "high")     fillClass = "risk-fill-high";
        if (riskClass === "moderate") fillClass = "risk-fill-moderate";

        const card = document.createElement("div");
        card.className = `abx-card risk-${riskClass}`;
        card.setAttribute("role", "article");

        card.innerHTML = `
            <div class="abx-card-header">
                <span class="abx-name">${p.antibiotic}</span>
                <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
                    <span class="abx-model-badge">${p.model_type}</span>
                    <span class="risk-badge ${badgeClass}">${p.risk_category} Risk</span>
                </div>
            </div>
            <div class="risk-bar-wrapper">
                <div class="risk-track" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100" aria-label="${pct}% resistance">
                    <div class="risk-fill ${fillClass}" style="width: ${pct}%"></div>
                </div>
                <span class="risk-pct">${pct}%</span>
            </div>
            <div class="abx-card-footer">
                <span class="abx-interp">${p.interpretation_label}</span>
                <button class="btn-explain" data-abx="${p.antibiotic}" aria-label="Explain ${p.antibiotic} prediction">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" aria-hidden="true">
                        <path d="M8.59 16.59L13.17 12 8.59 7.41 10 6l6 6-6 6z"/>
                    </svg>
                    Explain
                </button>
            </div>
        `;

        const explainBtn = card.querySelector(".btn-explain");
        explainBtn?.addEventListener("click", (e) => {
            e.stopPropagation();
            const abx = p.antibiotic;
            state.selectedAntibiotic = abx;

            const abxSelect = document.getElementById("explainAbxSelect");
            if (abxSelect) abxSelect.value = abx;
            document.getElementById("explainAbxName").textContent = abx;

            document.querySelectorAll(".abx-card").forEach(c =>
                c.classList.toggle("selected-for-explain", c === card)
            );

            updateShapExplanationView();
            document.getElementById("shapPanel")?.scrollIntoView({ behavior: "smooth", block: "start" });
        });

        grid.appendChild(card);
    });
}

// ─── Update SHAP View ────────────────────────────────────────────────────────
async function updateShapExplanationView() {
    if (!state.currentPatient) return;

    const abx             = state.selectedAntibiotic;
    const explainAbxName  = document.getElementById("explainAbxName");
    const explainProbText = document.getElementById("explainProbText");
    const explainRiskBadge= document.getElementById("explainRiskBadge");
    const posList         = document.getElementById("positiveFactorsList");
    const negList         = document.getElementById("negativeFactorsList");

    if (explainAbxName) explainAbxName.textContent = abx;

    try {
        const res = await fetch("/api/explain", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                patient_data: state.currentPatient,
                antibiotic: abx,
            }),
        });

        if (!res.ok) throw new Error("Failed to fetch SHAP explanation");

        const data = await res.json();

        const probPct = (data.estimated_resistance_probability * 100).toFixed(1);
        if (explainProbText) explainProbText.textContent = `${probPct}%`;

        let riskTier = "Low";
        let badgeClass = "risk-badge-low";
        if (data.estimated_resistance_probability > 0.65) {
            riskTier = "High"; badgeClass = "risk-badge-high";
        } else if (data.estimated_resistance_probability >= 0.35) {
            riskTier = "Moderate"; badgeClass = "risk-badge-moderate";
        }

        if (explainRiskBadge) {
            explainRiskBadge.className = `risk-badge ${badgeClass}`;
            explainRiskBadge.textContent = `${riskTier} Risk`;
        }

        const allFactors = [
            ...(data.top_positive_factors || []),
            ...(data.top_negative_factors || []),
        ];
        const maxShap = Math.max(...allFactors.map(f => Math.abs(f.shap_value)), 0.001);

        if (posList) {
            if (data.top_positive_factors?.length) {
                posList.innerHTML = data.top_positive_factors.map(f => {
                    const barWidth = Math.min(100, (Math.abs(f.shap_value) / maxShap) * 100).toFixed(1);
                    return `
                        <div class="shap-factor-item" role="listitem">
                            <div class="shap-factor-top">
                                <span class="shap-factor-name">${f.display_name}</span>
                                <span class="shap-factor-val pos">+${f.shap_value.toFixed(4)}</span>
                            </div>
                            <div class="shap-factor-bar-track">
                                <div class="shap-factor-bar-fill pos" style="width:${barWidth}%"></div>
                            </div>
                        </div>`;
                }).join("");
            } else {
                posList.innerHTML = `<div class="empty-factors">No significant risk-increasing factors.</div>`;
            }
        }

        if (negList) {
            if (data.top_negative_factors?.length) {
                negList.innerHTML = data.top_negative_factors.map(f => {
                    const barWidth = Math.min(100, (Math.abs(f.shap_value) / maxShap) * 100).toFixed(1);
                    return `
                        <div class="shap-factor-item" role="listitem">
                            <div class="shap-factor-top">
                                <span class="shap-factor-name">${f.display_name}</span>
                                <span class="shap-factor-val neg">${f.shap_value.toFixed(4)}</span>
                            </div>
                            <div class="shap-factor-bar-track">
                                <div class="shap-factor-bar-fill neg" style="width:${barWidth}%"></div>
                            </div>
                        </div>`;
                }).join("");
            } else {
                negList.innerHTML = `<div class="empty-factors">No significant risk-decreasing factors.</div>`;
            }
        }

        if (state.shapMode === "local") {
            renderPlotlyWaterfall(data);
        }

    } catch (err) {
        console.error("SHAP View Error:", err);
    }
}

// ─── Plotly Waterfall Chart (AMR Palette) ─────────────────────────────────────
function renderPlotlyWaterfall(data) {
    const container = document.getElementById("plotlyShapDiv");
    if (!container || !window.Plotly) return;

    const baseVal   = data.base_value;
    const finalProb = data.estimated_resistance_probability;
    const features  = data.waterfall_features || [];

    const labels   = ["Base Expected"];
    const deltas   = [baseVal];
    const measures = ["absolute"];
    const textVals = [`${(baseVal * 100).toFixed(1)}%`];

    features.forEach(f => {
        labels.push(f.display_name);
        deltas.push(f.shap_value);
        measures.push("relative");
        textVals.push(`${f.shap_value > 0 ? "+" : ""}${(f.shap_value * 100).toFixed(1)}%`);
    });

    labels.push("Final Prob");
    deltas.push(finalProb);
    measures.push("total");
    textVals.push(`${(finalProb * 100).toFixed(1)}%`);

    const plotData = [{
        type:        "waterfall",
        orientation: "v",
        measure:     measures,
        x:           labels,
        y:           deltas,
        text:        textVals,
        textposition:"outside",
        decreasing:  { marker: { color: "#8697C4" } },
        increasing:  { marker: { color: "#3D52A0" } },
        totals:      { marker: { color: "#7091E6" } },
        connector:   { line: { color: "#ADBBDA", width: 1, dash: "dot" } },
    }];

    const layout = {
        title: {
            text:  `SHAP Waterfall: ${data.antibiotic}`,
            font:  { color: "#3D52A0", size: 12.5, family: "Inter" },
        },
        paper_bgcolor: "transparent",
        plot_bgcolor:  "transparent",
        font:          { color: "#8697C4", family: "Inter", size: 10.5 },
        margin:        { l: 36, r: 16, t: 36, b: 110 },
        xaxis: {
            tickangle: -35,
            gridcolor: "rgba(173,187,218,0.3)",
            tickfont:  { color: "#6878A8", size: 9.5 },
        },
        yaxis: {
            title:     "Probability Impact",
            gridcolor: "rgba(173,187,218,0.3)",
            tickfont:  { color: "#6878A8" },
            range:     [0, Math.min(1.05, finalProb + 0.25)],
        },
        autosize: true,
    };

    Plotly.newPlot("plotlyShapDiv", plotData, layout, { responsive: true, displayModeBar: false });
}

// ─── Global SHAP Feature Importance Chart ─────────────────────────────────────
async function renderGlobalShapChart() {
    const abx = state.selectedAntibiotic;
    try {
        const res = await fetch(`/api/global-shap?antibiotic=${encodeURIComponent(abx)}`);
        if (!res.ok) throw new Error("Failed to load global SHAP summary");

        const data        = await res.json();
        const topFeatures = data.global_shap.top_features || [];

        const labels = topFeatures.map(f =>
            f.feature.replace(/_/g, " ").replace("organism ", "Pathogen: ").replace("infection type ", "Infection: ")
        );
        const values = topFeatures.map(f => f.mean_shap);

        labels.reverse();
        values.reverse();

        const plotData = [{
            type:       "bar",
            orientation:"h",
            x:          values,
            y:          labels,
            marker:     { color: "#7091E6" },
            text:       values.map(v => v.toFixed(4)),
            textposition:"auto",
            textfont:   { color: "#3D52A0", size: 10 },
        }];

        const layout = {
            title: {
                text: `Global Feature Importance (Mean |SHAP|): ${abx}`,
                font: { color: "#3D52A0", size: 12.5, family: "Inter" },
            },
            paper_bgcolor: "transparent",
            plot_bgcolor:  "transparent",
            font:          { color: "#8697C4", family: "Inter", size: 10.5 },
            margin:        { l: 150, r: 24, t: 36, b: 36 },
            xaxis: {
                title:     "Mean |SHAP Value|",
                gridcolor: "rgba(173,187,218,0.3)",
                tickfont:  { color: "#6878A8" },
            },
            yaxis: {
                gridcolor: "rgba(173,187,218,0.3)",
                tickfont:  { color: "#3D52A0", size: 9.5 },
            },
            autosize: true,
        };

        Plotly.newPlot("plotlyShapDiv", plotData, layout, { responsive: true, displayModeBar: false });

    } catch (err) {
        console.error("Global SHAP Error:", err);
    }
}

// ─── Database History & Persistence Handlers ─────────────────────────────────

// 1. Live Stats
async function loadDatabaseStats() {
    try {
        const res = await fetch("/api/stats");
        if (!res.ok) return;
        const stats = await res.json();

        const pEl = document.getElementById("metricTotalPreds");
        if (pEl) pEl.textContent = (stats.total_predictions || 0).toLocaleString();

        const uEl = document.getElementById("metricUniquePatients");
        if (uEl) uEl.textContent = (stats.unique_patients || 0).toLocaleString();

        const hEl = document.getElementById("metricHighRiskCount");
        if (hEl) hEl.textContent = (stats.high_risk_count || 0).toLocaleString();
    } catch (err) {
        console.error("Stats load error:", err);
    }
}

// 2. Recent Analyses List (Dashboard)
async function loadRecentAnalyses() {
    const container = document.getElementById("recentAnalysesList");
    if (!container) return;

    try {
        const res = await fetch("/api/history?limit=3");
        if (!res.ok) return;
        const data = await res.json();
        const items = data.history || [];

        if (!items.length) {
            container.innerHTML = `<div style="text-align:center; padding:16px; color:var(--text-secondary); font-size:12px;">No prediction records logged yet.</div>`;
            return;
        }

        container.innerHTML = items.map(item => {
            const riskClass = (item.highest_risk_category || "low").toLowerCase();
            const badgeClass = riskClass === "high" ? "risk-badge-high" : (riskClass === "moderate" ? "risk-badge-moderate" : "risk-badge-low");
            const pathogenName = (item.organism || "").replace(/_/g, " ");

            return `
                <div class="analysis-row">
                    <div class="analysis-row-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
                    </div>
                    <div class="analysis-row-body">
                        <div class="analysis-row-title">${item.patient_id} — ${pathogenName}</div>
                        <div class="analysis-row-meta">${item.created_at} · Peak: ${item.highest_risk_antibiotic} (${(item.highest_risk_prob * 100).toFixed(1)}%)</div>
                    </div>
                    <span class="risk-badge ${badgeClass}">${item.highest_risk_category} Risk</span>
                    <button class="btn btn-sm btn-secondary" onclick="loadPredictionIntoAnalyzer('${item.id}')" title="Inspect Prediction">View</button>
                </div>`;
        }).join("");

    } catch (err) {
        console.error("Recent analyses error:", err);
    }
}

// 3. Full Prediction History Table
async function loadPredictionHistory(search = "", risk = "All") {
    const tbody = document.getElementById("historyTableBody");
    if (!tbody) return;

    try {
        let url = `/api/history?limit=50`;
        if (search) url += `&search=${encodeURIComponent(search)}`;
        if (risk && risk !== "All") url += `&risk=${encodeURIComponent(risk)}`;

        const res = await fetch(url);
        if (!res.ok) throw new Error("Failed to fetch history records");
        const data = await res.json();
        const records = data.history || [];

        if (!records.length) {
            tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:32px; color:var(--text-secondary);">No matching prediction records found in SQLite database.</td></tr>`;
            return;
        }

        tbody.innerHTML = records.map(r => {
            const riskClass = (r.highest_risk_category || "low").toLowerCase();
            const badgeClass = riskClass === "high" ? "risk-badge-high" : (riskClass === "moderate" ? "risk-badge-moderate" : "risk-badge-low");
            const pathogenName = (r.organism || "").replace(/_/g, " ");
            const syndrome = (r.infection_type || "").replace(/_/g, " ");

            return `
                <tr>
                    <td><strong style="font-family:monospace; color:var(--amr-secondary);">${r.id}</strong></td>
                    <td><strong>${r.patient_id}</strong> (${r.age}y/${r.sex})</td>
                    <td>${pathogenName} <br><span style="font-size:11px; color:var(--text-secondary);">${syndrome}</span></td>
                    <td>
                        <span class="risk-badge ${badgeClass}">${r.highest_risk_category}</span>
                        <div style="font-size:11px; color:var(--text-secondary); margin-top:2px;">${r.highest_risk_antibiotic} (${(r.highest_risk_prob * 100).toFixed(1)}%)</div>
                    </td>
                    <td>${r.created_at}</td>
                    <td><span class="badge badge-info">${r.model_version}</span></td>
                    <td>
                        <div class="flex gap-3">
                            <button class="btn btn-sm btn-primary" onclick="loadPredictionIntoAnalyzer('${r.id}')" title="Load into Analyzer & SHAP">
                                Analyze
                            </button>
                            <button class="btn btn-sm btn-secondary" onclick="openReportModal('${r.id}')" title="Clinical Antibiogram Report">
                                Report
                            </button>
                        </div>
                    </td>
                </tr>`;
        }).join("");

    } catch (err) {
        console.error("History query error:", err);
        tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:24px; color:var(--text-secondary);">${err.message}</td></tr>`;
    }
}

// 4. Load Single Historical Record into Analyzer
window.loadPredictionIntoAnalyzer = async function(predictionId) {
    try {
        const res = await fetch(`/api/history/${predictionId}`);
        if (!res.ok) throw new Error("Record not found");
        const record = await res.json();

        if (record.input_data) {
            populateFormWithData(record.input_data);
            navigateTo("page-analyzer");
            
            // Execute prediction display
            const form = document.getElementById("patientForm");
            form?.dispatchEvent(new Event("submit"));
        }
    } catch (err) {
        console.error("Failed to load record:", err);
        alert(`Error loading analysis: ${err.message}`);
    }
};

// 5. Patient Directory
async function loadPatientsDirectory() {
    const tbody = document.getElementById("patientsTableBody");
    if (!tbody) return;

    try {
        const res = await fetch("/api/patients");
        if (!res.ok) throw new Error("Failed to load patients directory");
        const data = await res.json();
        const patients = data.patients || [];

        if (!patients.length) {
            tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding:32px; color:var(--text-secondary);">No patients recorded yet. Run a prediction to create records.</td></tr>`;
            return;
        }

        tbody.innerHTML = patients.map(p => {
            const riskClass = (p.last_risk_category || "low").toLowerCase();
            const badgeClass = riskClass === "high" ? "risk-badge-high" : (riskClass === "moderate" ? "risk-badge-moderate" : "risk-badge-low");
            const pathogen = (p.last_organism || "").replace(/_/g, " ");
            const syndrome = (p.last_infection_type || "").replace(/_/g, " ");

            return `
                <tr>
                    <td><strong style="font-family:monospace; color:var(--amr-primary);">${p.patient_id}</strong></td>
                    <td>${p.age} yrs / ${p.sex}</td>
                    <td><span class="badge badge-info">${p.total_analyses} analyses</span></td>
                    <td><strong>${pathogen}</strong></td>
                    <td>${syndrome}</td>
                    <td><span class="risk-badge ${badgeClass}">${p.last_risk_category} Risk</span></td>
                    <td>${p.last_analysis_date}</td>
                    <td>
                        <button class="btn btn-sm btn-primary" onclick="initNewAnalysisForPatient('${p.patient_id}', ${p.age}, '${p.sex}')">
                            New Analysis
                        </button>
                    </td>
                </tr>`;
        }).join("");

    } catch (err) {
        console.error("Patients load error:", err);
        tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding:24px; color:var(--text-secondary);">${err.message}</td></tr>`;
    }
}

window.initNewAnalysisForPatient = function(patientId, age, sex) {
    document.getElementById("inputPatientId").value = patientId;
    document.getElementById("inputAge").value = age;
    document.getElementById("inputSex").value = sex;
    navigateTo("page-analyzer");
};

// 6. Clinical Reports List
async function loadClinicalReports(search = "") {
    const grid = document.getElementById("reportsGrid");
    if (!grid) return;

    try {
        let url = "/api/reports";
        if (search) url += `?search=${encodeURIComponent(search)}`;

        const res = await fetch(url);
        if (!res.ok) throw new Error("Failed to load reports");
        const data = await res.json();
        const reports = data.reports || [];

        if (!reports.length) {
            grid.innerHTML = `<div style="grid-column:1/-1; text-align:center; padding:32px; color:var(--text-secondary);">No clinical reports found matching your criteria.</div>`;
            return;
        }

        grid.innerHTML = reports.map(r => {
            const riskClass = (r.highest_risk_category || "low").toLowerCase();
            const badgeClass = riskClass === "high" ? "risk-badge-high" : (riskClass === "moderate" ? "risk-badge-moderate" : "risk-badge-low");

            return `
                <div class="report-card">
                    <div class="report-card-top">
                        <span class="report-id">${r.report_id}</span>
                        <span class="risk-badge ${badgeClass}">${r.highest_risk_category} Risk</span>
                    </div>
                    <div class="report-title">${r.title}</div>
                    <div class="report-meta-row">
                        <span>Patient: <strong>${r.patient_id}</strong></span>
                        <span>Syndrome: ${r.infection_type}</span>
                    </div>
                    <div class="report-meta-row">
                        <span>Peak Resistance: <strong>${r.highest_risk_antibiotic}</strong></span>
                        <span>Prob: <strong>${(r.highest_risk_prob * 100).toFixed(1)}%</strong></span>
                    </div>
                    <div class="report-footer">
                        <span style="font-size:11px; color:var(--text-secondary);">${r.created_at}</span>
                        <button class="btn btn-sm btn-primary" onclick="openReportModal('${r.prediction_id}')">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                            Preview &amp; PDF
                        </button>
                    </div>
                </div>`;
        }).join("");

    } catch (err) {
        console.error("Reports load error:", err);
        grid.innerHTML = `<div style="grid-column:1/-1; text-align:center; padding:24px; color:var(--text-secondary);">${err.message}</div>`;
    }
}

// 7. Clinical Report Detail Modal Preview & PDF Print
window.openReportModal = async function(predictionId) {
    const modal = document.getElementById("reportModal");
    const body  = document.getElementById("reportBody");
    if (!modal || !body) return;

    body.innerHTML = `<div style="text-align:center; padding:40px; color:var(--text-secondary);">Generating clinical antibiogram report…</div>`;
    modal.classList.add("open");

    try {
        const res = await fetch(`/api/reports/${predictionId}`);
        if (!res.ok) throw new Error("Report not found");
        const r = await res.json();

        const p = r.patient_profile;
        const preds = r.antibiogram_predictions || [];

        body.innerHTML = `
            <div class="printable-report">
                <!-- Header Banner -->
                <div class="report-header-banner">
                    <div>
                        <div class="report-brand-name">AMR PREDICT — Clinical AI Report</div>
                        <div style="font-size:12px; color:var(--text-secondary); margin-top:2px;">Antimicrobial Resistance Decision-Support Antibiogram</div>
                    </div>
                    <div style="text-align:right;">
                        <div class="report-doc-type">${r.report_id}</div>
                        <div style="font-size:11px; color:var(--text-secondary); margin-top:2px;">Evaluated: ${r.analysis_date} UTC</div>
                    </div>
                </div>

                <!-- Patient Demographics & Labs Grid -->
                <div class="report-info-grid mb-3">
                    <div class="report-info-item">
                        <label>Patient Identifier</label>
                        <span>${r.patient_id}</span>
                    </div>
                    <div class="report-info-item">
                        <label>Demographics</label>
                        <span>${p.age} Years / ${p.sex === "F" ? "Female" : "Male"}</span>
                    </div>
                    <div class="report-info-item">
                        <label>Isolated Pathogen</label>
                        <span style="color:var(--amr-primary); font-weight:700;">${p.organism}</span>
                    </div>
                    <div class="report-info-item">
                        <label>Infection Syndrome</label>
                        <span>${p.infection_type}</span>
                    </div>
                    <div class="report-info-item">
                        <label>Serum Creatinine</label>
                        <span>${p.creatinine_mg_dl} mg/dL</span>
                    </div>
                    <div class="report-info-item">
                        <label>WBC Count</label>
                        <span>${p.wbc_count_k_ul} k/µL</span>
                    </div>
                </div>

                <!-- Highest Risk Callout Box -->
                <div class="notice-box mb-3" style="background:rgba(61,82,160,0.06); border-color:var(--amr-secondary);">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/></svg>
                    <div>
                        <strong>Highest Estimated Non-Susceptibility:</strong> ${r.highest_risk_summary.antibiotic} (${(r.highest_risk_summary.probability * 100).toFixed(1)}% — ${r.highest_risk_summary.risk_category} Risk Category).
                    </div>
                </div>

                <!-- Antibiogram Prediction Matrix Table -->
                <div style="font-size:13px; font-weight:700; color:var(--amr-primary); margin-bottom:6px;">Multi-Agent Resistance Estimation Matrix</div>
                <div class="table-wrapper mb-3">
                    <table class="clinical-table">
                        <thead>
                            <tr>
                                <th>Target Antibiotic</th>
                                <th>Model Architecture</th>
                                <th>Resistance Probability</th>
                                <th>Risk Tier</th>
                                <th>Clinical Interpretation</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${preds.map(pred => {
                                const rk = (pred.risk_category || "").toLowerCase();
                                const bClass = rk === "high" ? "risk-badge-high" : (rk === "moderate" ? "risk-badge-moderate" : "risk-badge-low");
                                const pct = (pred.estimated_resistance_probability * 100).toFixed(1);
                                return `
                                    <tr>
                                        <td><strong>${pred.antibiotic}</strong></td>
                                        <td><span class="badge badge-info">${pred.model_type}</span></td>
                                        <td><strong>${pct}%</strong></td>
                                        <td><span class="risk-badge ${bClass}">${pred.risk_category}</span></td>
                                        <td style="font-size:11.5px; color:var(--text-secondary);">${pred.interpretation_label}</td>
                                    </tr>`;
                            }).join("")}
                        </tbody>
                    </table>
                </div>

                <!-- Attribution & Legal Disclaimer -->
                <div class="disclaimer-box" style="margin-bottom:0; font-size:11px;">
                    <strong>Model Attribution &amp; Disclaimer:</strong> Predictions represent statistical estimations generated by AMR-X v1.0. This prototype does not provide prescription or dosage recommendations. Clinical microbiological culturing &amp; standard AST remains mandatory.
                </div>
            </div>`;

    } catch (err) {
        console.error("Failed to render report:", err);
        body.innerHTML = `<div style="text-align:center; padding:32px; color:var(--text-secondary);">${err.message}</div>`;
    }
};

// ─── Load Model Benchmarks ────────────────────────────────────────────────────
async function loadModelBenchmarks() {
    try {
        const res = await fetch("/api/models");
        if (!res.ok) throw new Error("Failed to load model benchmarks");

        const data   = await res.json();
        state.modelsData = data;

        const tbody = document.getElementById("modelsTableBody");
        if (tbody && data.registry) {
            tbody.innerHTML = Object.entries(data.registry).map(([abx, meta]) => {
                const m    = meta.metrics;
                const prev = (meta.prevalence * 100).toFixed(1);
                return `
                    <tr>
                        <td><strong>${abx}</strong></td>
                        <td>${prev}%</td>
                        <td><span class="badge badge-info">${meta.model_type}</span></td>
                        <td><span class="auc-highlight">${m.roc_auc.toFixed(3)}</span></td>
                        <td>${m.f1.toFixed(3)}</td>
                        <td>${m.recall.toFixed(3)}</td>
                        <td>${m.precision.toFixed(3)}</td>
                        <td>${m.accuracy.toFixed(3)}</td>
                        <td>${m.brier_score.toFixed(3)}</td>
                    </tr>`;
            }).join("");
        }

        renderRocCurves(data.detailed_metrics);
        renderCalibrationCurves(data.detailed_metrics);

    } catch (err) {
        console.error("Benchmarks Load Error:", err);
    }
}

// ─── ROC Curves Chart ─────────────────────────────────────────────────────────
function renderRocCurves(detailedMetrics) {
    const ctx = document.getElementById("rocChartCanvas");
    if (!ctx || !window.Chart || !detailedMetrics) return;

    const datasets = [];
    let colorIdx = 0;

    Object.entries(detailedMetrics).forEach(([abx, item]) => {
        const candidate = item.candidates[item.selected_model];
        if (candidate?.roc_curve) {
            const { fpr, tpr } = candidate.roc_curve;
            datasets.push({
                label:       `${abx} (AUC: ${candidate.roc_auc.toFixed(3)})`,
                data:        fpr.map((x, i) => ({ x, y: tpr[i] })),
                borderColor: AMR_PALETTE[colorIdx % AMR_PALETTE.length],
                borderWidth: 2,
                pointRadius: 0,
                fill:        false,
                tension:     0.1,
            });
            colorIdx++;
        }
    });

    datasets.push({
        label:       "Chance Baseline (AUC: 0.500)",
        data:        [{ x: 0, y: 0 }, { x: 1, y: 1 }],
        borderColor: "rgba(173,187,218,0.5)",
        borderWidth: 1.5,
        borderDash:  [4, 4],
        pointRadius: 0,
        fill:        false,
    });

    state.chartInstances["rocChartCanvas"]?.destroy();
    state.chartInstances["rocChartCanvas"] = new Chart(ctx, {
        type: "line",
        data: { datasets },
        options: {
            responsive:         true,
            maintainAspectRatio:false,
            scales: {
                x: {
                    type:   "linear",
                    min: 0, max: 1,
                    title:  { display: true, text: "False Positive Rate", color: "#8697C4", font: { size: 11 } },
                    grid:   { color: "rgba(173,187,218,0.25)" },
                    ticks:  { color: "#8697C4", font: { size: 10 } },
                },
                y: {
                    type:   "linear",
                    min: 0, max: 1,
                    title:  { display: true, text: "True Positive Rate", color: "#8697C4", font: { size: 11 } },
                    grid:   { color: "rgba(173,187,218,0.25)" },
                    ticks:  { color: "#8697C4", font: { size: 10 } },
                },
            },
            plugins: {
                legend: {
                    labels:   { color: "#3D52A0", font: { size: 9.5 } },
                    position: "bottom",
                },
            },
        },
    });
}

// ─── Calibration Curves Chart ─────────────────────────────────────────────────
function renderCalibrationCurves(detailedMetrics) {
    const ctx = document.getElementById("calibChartCanvas");
    if (!ctx || !window.Chart || !detailedMetrics) return;

    const datasets = [];
    let colorIdx = 0;

    Object.entries(detailedMetrics).forEach(([abx, item]) => {
        const candidate = item.candidates[item.selected_model];
        if (candidate?.calibration?.prob_pred) {
            const { prob_pred, prob_true } = candidate.calibration;
            datasets.push({
                label:       `${abx} (Brier: ${candidate.brier_score.toFixed(3)})`,
                data:        prob_pred.map((x, i) => ({ x, y: prob_true[i] })),
                borderColor: AMR_PALETTE[colorIdx % AMR_PALETTE.length],
                borderWidth: 2,
                pointRadius: 2.5,
                pointBackgroundColor: AMR_PALETTE[colorIdx % AMR_PALETTE.length],
                fill:        false,
            });
            colorIdx++;
        }
    });

    datasets.push({
        label:       "Perfect Calibration",
        data:        [{ x: 0, y: 0 }, { x: 1, y: 1 }],
        borderColor: "rgba(173,187,218,0.5)",
        borderWidth: 1.5,
        borderDash:  [4, 4],
        pointRadius: 0,
        fill:        false,
    });

    state.chartInstances["calibChartCanvas"]?.destroy();
    state.chartInstances["calibChartCanvas"] = new Chart(ctx, {
        type: "line",
        data: { datasets },
        options: {
            responsive:         true,
            maintainAspectRatio:false,
            scales: {
                x: {
                    type:   "linear",
                    min: 0, max: 1,
                    title:  { display: true, text: "Predicted Probability", color: "#8697C4", font: { size: 11 } },
                    grid:   { color: "rgba(173,187,218,0.25)" },
                    ticks:  { color: "#8697C4", font: { size: 10 } },
                },
                y: {
                    type:   "linear",
                    min: 0, max: 1,
                    title:  { display: true, text: "Observed Fraction", color: "#8697C4", font: { size: 11 } },
                    grid:   { color: "rgba(173,187,218,0.25)" },
                    ticks:  { color: "#8697C4", font: { size: 10 } },
                },
            },
            plugins: {
                legend: {
                    labels:   { color: "#3D52A0", font: { size: 9.5 } },
                    position: "bottom",
                },
            },
        },
    });
}

// ─── Dataset Insights (EDA) ───────────────────────────────────────────────────
async function loadDatasetEda() {
    try {
        const res = await fetch("/api/eda");
        if (!res.ok) throw new Error("Failed to load EDA data");

        const data   = await res.json();
        state.edaData = data;

        if (data.dataset_metadata) {
            const el1 = document.getElementById("kpiPatients");
            if (el1) el1.textContent = data.dataset_metadata.total_records.toLocaleString();
        }
        if (data.age_distribution) {
            const el = document.getElementById("kpiAgeMean");
            if (el) el.textContent = `${data.age_distribution.mean} yrs`;
        }

        renderEdaResistanceChart(data.resistance_prevalence);
        renderEdaOrganismChart(data.organism_distribution);
        renderEdaInfectionChart(data.infection_distribution);
        renderEdaComorbiditiesChart(data.clinical_risk_factors_prevalence);

    } catch (err) {
        console.error("EDA Load Error:", err);
    }
}

function renderEdaResistanceChart(resPrev) {
    const ctx = document.getElementById("edaResistanceChart");
    if (!ctx || !window.Chart || !resPrev) return;

    const labels = Object.keys(resPrev);
    const values = Object.values(resPrev).map(v => (v.resistance_rate * 100).toFixed(1));

    state.chartInstances["edaResistanceChart"]?.destroy();
    state.chartInstances["edaResistanceChart"] = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label:           "Resistance Prevalence (%)",
                data:            values,
                backgroundColor: labels.map((_, i) => AMR_PALETTE_ALPHA(AMR_PALETTE[i % AMR_PALETTE.length], 0.72)),
                borderColor:     labels.map((_, i) => AMR_PALETTE[i % AMR_PALETTE.length]),
                borderWidth:     1.5,
                borderRadius:    5,
            }],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                x: { ticks: { color: "#8697C4", font: { size: 9.5 } }, grid: { display: false } },
                y: { min: 0, max: 100, ticks: { color: "#8697C4", font: { size: 10 } }, grid: { color: "rgba(173,187,218,0.25)" } },
            },
            plugins: { legend: { display: false } },
        },
    });
}

function renderEdaOrganismChart(orgDist) {
    const ctx = document.getElementById("edaOrganismChart");
    if (!ctx || !window.Chart || !orgDist) return;

    const labels = Object.keys(orgDist).map(k => k.replace(/_/g, " "));
    const values = Object.values(orgDist);

    state.chartInstances["edaOrganismChart"]?.destroy();
    state.chartInstances["edaOrganismChart"] = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels,
            datasets: [{
                data:            values,
                backgroundColor: AMR_PALETTE.slice(0, labels.length),
                borderColor:     "#FFFFFF",
                borderWidth:     2,
            }],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { position: "right", labels: { color: "#3D52A0", font: { size: 10.5 } } },
            },
        },
    });
}

function renderEdaInfectionChart(infDist) {
    const ctx = document.getElementById("edaInfectionChart");
    if (!ctx || !window.Chart || !infDist) return;

    const labels = Object.keys(infDist).map(k => k.replace(/_/g, " "));
    const values = Object.values(infDist);

    state.chartInstances["edaInfectionChart"]?.destroy();
    state.chartInstances["edaInfectionChart"] = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label:           "Patient Count",
                data:            values,
                backgroundColor: AMR_PALETTE_ALPHA("#7091E6", 0.72),
                borderColor:     "#7091E6",
                borderWidth:     1.5,
                borderRadius:    5,
            }],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                x: { ticks: { color: "#8697C4", font: { size: 9.5 } }, grid: { display: false } },
                y: { ticks: { color: "#8697C4", font: { size: 10 } }, grid: { color: "rgba(173,187,218,0.25)" } },
            },
            plugins: { legend: { display: false } },
        },
    });
}

function renderEdaComorbiditiesChart(comorb) {
    const ctx = document.getElementById("edaComorbiditiesChart");
    if (!ctx || !window.Chart || !comorb) return;

    const labels = Object.keys(comorb);
    const values = Object.values(comorb).map(v => (v * 100).toFixed(1));

    state.chartInstances["edaComorbiditiesChart"]?.destroy();
    state.chartInstances["edaComorbiditiesChart"] = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label:           "Prevalence in Cohort (%)",
                data:            values,
                backgroundColor: AMR_PALETTE_ALPHA("#8697C4", 0.72),
                borderColor:     "#8697C4",
                borderWidth:     1.5,
                borderRadius:    4,
            }],
        },
        options: {
            indexAxis:          "y",
            responsive:         true,
            maintainAspectRatio:false,
            scales: {
                x: {
                    min: 0, max: 100,
                    ticks: { color: "#8697C4", font: { size: 10 } },
                    grid:  { color: "rgba(173,187,218,0.25)" },
                },
                y: {
                    ticks: { color: "#3D52A0", font: { size: 9.5 } },
                    grid:  { display: false },
                },
            },
            plugins: { legend: { display: false } },
        },
    });
}

// ─── Utilities ──────────────────────────────────────────────────────────────
function resizeCharts(canvasIds) {
    canvasIds.forEach(id => state.chartInstances[id]?.resize());
}

function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

const spinStyle = document.createElement("style");
spinStyle.textContent = `@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`;
document.head.appendChild(spinStyle);
