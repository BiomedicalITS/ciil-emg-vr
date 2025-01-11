using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GeneralManager : MonoBehaviour
{
    public int NumReps;
    public float TimePerRep;
    public float TimeBetRep;
    public string InputsNames;
    public string OutputFolder;

    private static GeneralManager instance;

    void Awake()
    {
        if (instance == null)
        {
            instance = this;
            DontDestroyOnLoad(this.gameObject);
        }
        else
        {
            Destroy(this.gameObject);
        }
    }

    void Start()
    {
        // Initialization if needed
    }

    void Update()
    {
        // Update logic if needed
    }

    public void InitialTrainingConfig(string message)
    {
        // Split the message to extract the individual configuration values
        string[] parts = message.Split(' ');

        // Parse and store the configuration values
        NumReps = int.Parse(parts[1]);
        TimePerRep = float.Parse(parts[2]);
        TimeBetRep = float.Parse(parts[3]);
        InputsNames = parts[4];
        OutputFolder = parts[5];

        // Call the function to activate HandGesturesProp
        ActivateHandGesturesProp();
    }

    private void ActivateHandGesturesProp()
    {
        // Find the HandGestureFeedback game object
        GameObject handGestureFeedback = GameObject.Find("HandGestureFeedback");
        if (handGestureFeedback == null)
        {
            Debug.LogError("HandGestureFeedback game object not found");
            return;
        }

        // Split the InputsNames into individual names
        string[] inputNamesArray = InputsNames.Split(',');

        // Get the HandGestureActivation component
        HandGestureActivation handGestureActivation = handGestureFeedback.GetComponent<HandGestureActivation>();
        if (handGestureActivation == null)
        {
            Debug.LogError("HandGestureActivation component not found on HandGestureFeedback");
            return;
        }

        // Iterate over each input name
        foreach (string inputName in inputNamesArray)
        {
            // Remove the ".png" suffix from the input name
            string cleanInputName = inputName.Trim().Replace(".png", "");

            // Activate the hand gesture prop
            handGestureActivation.Activate(cleanInputName);
        }
    }
}
