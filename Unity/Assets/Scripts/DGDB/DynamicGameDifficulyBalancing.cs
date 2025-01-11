using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DynamicGameDifficultyBalancing : MonoBehaviour
{
    // Game parameters
    public float DifficultyOfMovements;
    public float ClassificationThreshold;
    public float PositionAnglesObjects;
    public float DifficultyLevel;

    // Measurements
    public float CompletionTime;
    public float Accuracy;
    public float ErrorRate;
    public float FrequencyInitialSuccess;
    private float Score;
    public float UsersFatigue;
    public float AdjustedScore;

    // GameObject reference
    GameObject emgObject;

    // Start is called before the first frame update
    void Start()
    {
        // Move GameObject.Find to Start() to ensure it runs after object instantiation
        emgObject = GameObject.Find("EMGRawReader");
        if (emgObject == null)
        {
            Debug.LogError("EMGRawReader not found. Make sure the object exists in the scene.");
        }
    }

    void HandleAccuracy()
    {
        // Check if emgObject is null before trying to access its component
        if (emgObject != null)
        {
            Accuracy = emgObject.GetComponent<EMGRawReader>().EMG_velocity;
        }
        else
        {
            Debug.LogError("EMGRawReader is null. Ensure it has been assigned in the inspector.");
        }
    }

    // Update is called once per frame
    void Update()
    {
        HandleAccuracy();
        // Add your dynamic difficulty balancing logic here based on the collected measurements.
    }
}
