-- Table: energy.verblijfsobjectgebruiksdoelen

-- DROP TABLE energy.verblijfsobjectgebruiksdoelen;

CREATE TABLE energy.verblijfsobjectgebruiksdoelen AS
SELECT
  vogdab.identificatie,
  array_agg(vogdab.gebruiksdoelverblijfsobject) AS gebruiksdoelen
FROM
  bagactueel.verblijfsobjectgebruiksdoelactueelbestaand vogdab
GROUP BY
  vogdab.identificatie;

ALTER TABLE energy.verblijfsobjectgebruiksdoelen
  OWNER TO postgres;

-- Index: energy.verblijfsobjectgebruiksdoelen_identificatie_idx

-- DROP INDEX energy.verblijfsobjectgebruiksdoelen_identificatie_idx;

CREATE INDEX verblijfsobjectgebruiksdoelen_identificatie_idx
  ON energy.verblijfsobjectgebruiksdoelen
  USING btree
  (identificatie COLLATE pg_catalog."default");

