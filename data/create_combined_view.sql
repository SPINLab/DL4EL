DROP VIEW energy.combined_buildings_energy_labels;

CREATE OR REPLACE VIEW energy.combined_buildings_energy_labels AS
 SELECT bep."POSTCODE_WONING" AS postal_code,
    naab.huisnummer AS house_number,
    bep."HUISNUMMER_TOEV_WONING" AS house_number_addition,
    vbogdab.gebruiksdoelen AS purposes,
    pandab.bouwjaar AS year_of_construction,
    replace(bep."EP_TXT"::text, ','::text, '.'::text)::numeric AS energy_performance_index,
    bep."LABEL" AS energy_performance_label,
    bep."OPNAMEDATUM" AS recorded_date,
    bep."REGISTRATIEDATUM" AS registration_date,
    st_astext(st_snaptogrid(st_transform(st_force2d(pandab.geovlak), 4326), 0.000001::double precision)) AS geometry_wgs84,
    st_astext(st_snaptogrid(st_centroid(st_transform(st_force2d(pandab.geovlak), 4326)), 0.000001::double precision)) AS centroid_wgs84
   FROM energy.building_energy_performance bep
     JOIN bagactueel.nummeraanduidingactueelbestaand naab ON bep."POSTCODE_WONING"::text = naab.postcode::text AND bep."HUISNUMMER_WONING" = naab.huisnummer AND (lower(bep."HUISNUMMER_TOEV_WONING"::text) = lower(concat(naab.huisletter, naab.huisnummertoevoeging)) OR lower(bep."HUISNUMMER_TOEV_WONING"::text) = lower(concat(naab.huisnummertoevoeging, naab.huisletter)) OR bep."HUISNUMMER_TOEV_WONING" IS NULL AND concat(naab.huisletter, naab.huisnummertoevoeging) = ''::text)
     JOIN bagactueel.adres adres ON naab.identificatie::text = adres.nummeraanduiding::text
     JOIN bagactueel.verblijfsobjectpandactueelbestaand vbopab ON adres.adresseerbaarobject::text = vbopab.identificatie::text
     JOIN energy.verblijfsobjectgebruiksdoelen vbogdab ON adres.adresseerbaarobject::text = vbogdab.identificatie::text
     JOIN bagactueel.pandactueelbestaand pandab ON vbopab.gerelateerdpand::text = pandab.identificatie::text
  WHERE bep."LABEL" IS NOT NULL;
--  GROUP BY bep."POSTCODE_WONING", naab.huisnummer, bep."HUISNUMMER_TOEV_WONING", pandab.bouwjaar, (replace(bep."EP_TXT"::text, ','::text, '.'::text)::numeric), bep."LABEL", bep."OPNAMEDATUM", bep."REGISTRATIEDATUM", (st_astext(st_snaptogrid(st_transform(st_force2d(pandab.geovlak), 4326), 0.000001::double precision))), (st_astext(st_snaptogrid(st_centroid(st_transform(st_force2d(pandab.geovlak), 4326)), 0.000001::double precision)));

ALTER TABLE energy.combined_buildings_energy_labels
  OWNER TO postgres;
