-- Create stored procedure ComputeAverageWeightedScoreForUser
DELIMITER $$

CREATE PROCEDURE ComputeAverageWeightedScoreForUser(
    IN user_id INT
)
BEGIN
    DECLARE weighted_avg FLOAT;
    
    -- Compute weighted average score for the user
    SELECT SUM(corrections.score * projects.weight) / SUM(projects.weight)
    INTO weighted_avg
    FROM corrections
    INNER JOIN projects ON corrections.project_id = projects.id
    WHERE corrections.user_id = user_id;
    
    -- Update the user's average_score in the users table
    UPDATE users
    SET average_score = weighted_avg
    WHERE id = user_id;
    
END$$

DELIMITER ;