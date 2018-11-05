-- View: energy.combined_buildings_energy_labels

CREATE OR REPLACE VIEW energy.combined_buildings_energy_labels AS
SELECT bep."POSTCODE_WONING" AS postal_code,
  naab.huisnummer AS house_number,
  bep."HUISNUMMER_TOEV_WONING" AS house_number_addition,
  array_to_json(vbogdab.gebruiksdoelen) AS purposes,
  pandab.bouwjaar AS year_of_construction,
  cast(replace(bep."EP_TXT", ',', '.') as numeric) AS energy_performance_index,
  bep."LABEL" AS energy_performance_label,
  bep."OPNAMEDATUM" AS recorded_date,
  bep."REGISTRATIEDATUM" AS registration_date,
  st_astext(st_snaptogrid(st_transform(st_force2d(pandab.geovlak), 4326), 0.000001)) AS geometry_wgs84,
  st_astext(st_snaptogrid(st_centroid(st_transform(st_force2d(pandab.geovlak), 4326)), 0.000001)) AS centroid_wgs84,
  st_astext(st_snaptogrid(st_transform(st_force2d(st_collect(array(
    SELECT pab2.geovlak FROM bagactueel.pandactueelbestaand pab2
    WHERE st_dwithin(pandab.geovlak, pab2.geovlak, 50)
  ))), 4326), 0.000001)) AS g2
  FROM energy.building_energy_performance bep
    JOIN bagactueel.nummeraanduidingactueelbestaand naab ON
      bep."POSTCODE_WONING" = naab.postcode
      AND bep."HUISNUMMER_WONING" = naab.huisnummer
      AND ( -- match if:
        lower(bep."HUISNUMMER_TOEV_WONING") = lower(concat(naab.huisletter, naab.huisnummertoevoeging)) -- it is either exactly the same as huisletter and huisnummertoevoeging
        OR lower(bep."HUISNUMMER_TOEV_WONING") = lower(concat(naab.huisnummertoevoeging, naab.huisletter)) -- or exactly the same as the reverse
        OR (bep."HUISNUMMER_TOEV_WONING" IS NULL AND concat(naab.huisletter, naab.huisnummertoevoeging) = '') -- or they are both empty
      )
    JOIN bagactueel.adres adres ON naab.identificatie = adres.nummeraanduiding
    JOIN bagactueel.verblijfsobjectpandactueelbestaand vbopab ON adres.adresseerbaarobject = vbopab.identificatie
    JOIN energy.verblijfsobjectgebruiksdoelen vbogdab ON adres.adresseerbaarobject = vbogdab.identificatie
    JOIN bagactueel.pandactueelbestaand pandab ON vbopab.gerelateerdpand = pandab.identificatie
  WHERE bep."LABEL" IS NOT NULL;

ALTER TABLE energy.combined_buildings_energy_labels
  OWNER TO postgres;
