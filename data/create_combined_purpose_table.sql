-- Table: energy.verblijfsobjectgebruiksdoelen

-- DROP TABLE energy.verblijfsobjectgebruiksdoelen;

CREATE TABLE energy.verblijfsobjectgebruiksdoelen
(
  identificatie character varying(16),
  gebruiksdoelen bagactueel.gebruiksdoelverblijfsobject[]
)
WITH (
  OIDS=FALSE
);
ALTER TABLE energy.verblijfsobjectgebruiksdoelen
  OWNER TO postgres;

-- Index: energy.verblijfsobjectgebruiksdoelen_identificatie_idx

-- DROP INDEX energy.verblijfsobjectgebruiksdoelen_identificatie_idx;

CREATE INDEX verblijfsobjectgebruiksdoelen_identificatie_idx
  ON energy.verblijfsobjectgebruiksdoelen
  USING btree
  (identificatie COLLATE pg_catalog."default");

