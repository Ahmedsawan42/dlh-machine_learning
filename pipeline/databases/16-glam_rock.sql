-- List all Glam rock bands ranked by their lifespan until 2020
SELECT 
    band_name,
    (COALESCE(split, 2020) - formed) AS lifespan
FROM metal_bands
WHERE style LIKE '%Glam rock%'
ORDER BY lifespan DESC;