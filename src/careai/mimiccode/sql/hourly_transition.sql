WITH hourly_base AS (
    SELECT
        ie.subject_id,
        ie.hadm_id,
        ih.stay_id,
        ih.hr,
        (ih.endtime - INTERVAL '1 hour') AS starttime,
        ih.endtime
    FROM mimiciv_derived.icustay_hourly AS ih
    INNER JOIN mimiciv_icu.icustays AS ie
        ON ih.stay_id = ie.stay_id
    INNER JOIN mimiciv_derived.icustay_detail AS idet
        ON ih.stay_id = idet.stay_id
    WHERE ih.hr >= 0
      AND idet.admission_age >= 18
      /*__STAY_FILTER__*/
),
state_hourly AS (
    SELECT
        hb.subject_id,
        hb.hadm_id,
        hb.stay_id,
        hb.hr,
        hb.starttime,
        hb.endtime,
        sf.sofa_24hours AS s_t_sofa,
        AVG(vs.mbp) AS s_t_mbp,
        AVG(vs.heart_rate) AS s_t_heart_rate,
        AVG(vs.resp_rate) AS s_t_resp_rate,
        AVG(vs.spo2) AS s_t_spo2,
        MAX(gcs.gcs) AS s_t_gcs,
        MAX(uo.uo_mlkghr_24hr) AS s_t_urine_output_rate,
        AVG(od.o2_flow) AS s_t_oxygen_delivery,
        AVG(ch.creatinine) AS s_t_creatinine,
        AVG(ch.bun) AS s_t_bun,
        MAX(idet.admission_age) AS s_t_age,
        MAX(chs.charlson_comorbidity_index) AS s_t_charlson
    FROM hourly_base AS hb
    LEFT JOIN mimiciv_derived.sofa AS sf
        ON hb.stay_id = sf.stay_id
       AND hb.hr = sf.hr
    LEFT JOIN mimiciv_derived.vitalsign AS vs
        ON hb.stay_id = vs.stay_id
       AND hb.starttime < vs.charttime
       AND hb.endtime >= vs.charttime
    LEFT JOIN mimiciv_derived.gcs AS gcs
        ON hb.stay_id = gcs.stay_id
       AND hb.starttime < gcs.charttime
       AND hb.endtime >= gcs.charttime
    LEFT JOIN mimiciv_derived.urine_output_rate AS uo
        ON hb.stay_id = uo.stay_id
       AND hb.starttime < uo.charttime
       AND hb.endtime >= uo.charttime
    LEFT JOIN mimiciv_derived.oxygen_delivery AS od
        ON hb.stay_id = od.stay_id
       AND hb.starttime < od.charttime
       AND hb.endtime >= od.charttime
    LEFT JOIN mimiciv_derived.chemistry AS ch
        ON hb.hadm_id = ch.hadm_id
       AND hb.starttime < ch.charttime
       AND hb.endtime >= ch.charttime
    LEFT JOIN mimiciv_derived.icustay_detail AS idet
        ON hb.stay_id = idet.stay_id
    LEFT JOIN mimiciv_derived.charlson AS chs
        ON hb.hadm_id = chs.hadm_id
    GROUP BY
        hb.subject_id, hb.hadm_id, hb.stay_id, hb.hr, hb.starttime, hb.endtime, sf.sofa_24hours
),
actions_hourly AS (
    SELECT
        hb.stay_id,
        hb.hr,
        MAX(
            CASE WHEN va.stay_id IS NOT NULL THEN 1 ELSE 0 END
        ) AS a_t_vaso,
        MAX(
            CASE
                WHEN vt.stay_id IS NOT NULL
                 AND vt.ventilation_status IS NOT NULL
                 AND vt.ventilation_status <> 'None'
                THEN 1 ELSE 0
            END
        ) AS a_t_vent,
        MAX(
            CASE
                WHEN cr.stay_id IS NOT NULL
                 AND (cr.system_active = 1 OR cr.crrt_mode IS NOT NULL)
                THEN 1 ELSE 0
            END
        ) AS a_t_crrt
    FROM hourly_base AS hb
    LEFT JOIN mimiciv_derived.vasoactive_agent AS va
        ON hb.stay_id = va.stay_id
       AND hb.endtime > va.starttime
       AND hb.endtime <= va.endtime
    LEFT JOIN mimiciv_derived.ventilation AS vt
        ON hb.stay_id = vt.stay_id
       AND hb.endtime > vt.starttime
       AND hb.endtime <= vt.endtime
    LEFT JOIN mimiciv_derived.crrt AS cr
        ON hb.stay_id = cr.stay_id
       AND hb.starttime < cr.charttime
       AND hb.endtime >= cr.charttime
    GROUP BY hb.stay_id, hb.hr
)
SELECT
    sh.subject_id,
    sh.hadm_id,
    sh.stay_id,
    sh.hr,
    sh.starttime,
    sh.endtime,
    sh.s_t_sofa,
    sh.s_t_mbp,
    sh.s_t_heart_rate,
    sh.s_t_resp_rate,
    sh.s_t_spo2,
    sh.s_t_gcs,
    sh.s_t_urine_output_rate,
    sh.s_t_oxygen_delivery,
    sh.s_t_creatinine,
    sh.s_t_bun,
    sh.s_t_age,
    sh.s_t_charlson,
    COALESCE(ah.a_t_vaso, 0) AS a_t_vaso,
    COALESCE(ah.a_t_vent, 0) AS a_t_vent,
    COALESCE(ah.a_t_crrt, 0) AS a_t_crrt
FROM state_hourly AS sh
LEFT JOIN actions_hourly AS ah
    ON sh.stay_id = ah.stay_id
   AND sh.hr = ah.hr
ORDER BY sh.stay_id, sh.hr;
