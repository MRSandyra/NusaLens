CREATE OR REPLACE VIEW v_indicator_series AS
SELECT
    dc.name_id AS country_name,
    di.name_id AS indicator_name,
    fi.year,
    fi.value,
    di.unit
FROM fact_indicator fi
JOIN dim_country dc ON dc.iso3 = fi.iso3
JOIN dim_indicator di ON di.code = fi.indicator_code;
