using System;
using System.IO;
using UnityEngine;

public class PositionSaver : MonoBehaviour
{
    public int rate = 60; // Number of frames between position saves
    public GameObject[] objectsToSave; // List of GameObjects whose positions will be saved
    public string SavePath = ""; // The path where the file will be saved
    public TimerController timerController; // Reference to the TimerController script
    public bool saveOnReset = true; // Save positions on timer reset if true, otherwise don't save

    public Transform _camera; // Reference to the camera transform
    public Transform _hand; // Reference to the hand transform

    public OnlyOneGrabAtATime _grab; // Reference to the OnlyOneGrabAtATime script

    private int frameCount = 0; // Counter to track the number of frames
    private StreamWriter fileWriter; // StreamWriter for writing to the file
    private bool savePositions = false; // Indicates whether to save positions

    private void Start()
    {
        // Initialize references to objects and components
        GameObject _objects = GameObject.Find("NewObjects v2");
        GameObject _apple = GameObject.Find("NewObjects v2/Apple");
        _grab = _objects.GetComponent<OnlyOneGrabAtATime>();

        // Initialize PlayerPrefs for trial times
        PlayerPrefs.SetString("Trialtimes", "");

        // Subscribe to timer events if the timerController is assigned
        if (timerController != null)
        {
            timerController.OnTimerStart += OnTimerStart;
            timerController.OnTimerReset += OnTimerReset;
            timerController.OnTimerComplete += OnTimerComplete;
        }

        // Example of setting up the file writer and header row
        // Generate the filename based on the current time
        // string fileName = "positions_" + System.DateTime.Now.ToString("yyyy_MM_dd_HH_mm_ss") + ".txt";
        // string filePath = Path.Combine(SavePath, fileName);

        // Create the directory if it doesn't exist
        // Directory.CreateDirectory(SavePath);

        // Open the file for writing
        // fileWriter = new StreamWriter(filePath);

        // Write the header row with object names
        // fileWriter.Write("Timestamp"); // Adding the timestamp column header

        // for (int i = 0; i < objectsToSave.Length; i++)
        // {
        //     GameObject obj = objectsToSave[i];
        //     fileWriter.Write("\t" + obj.name); // Use tab as a separator for columns
        // }

        // fileWriter.WriteLine(); // Move to the next line after writing the header row
    }

    private void Update()
    {
        // Save positions if the timer is running and savePositions is true
        if (timerController != null && timerController.IsTimerRunning && savePositions)
        {
            frameCount++;

            // Save the positions at every "rate" frame
            if (frameCount >= rate)
            {
                SavePositions();
                frameCount = 0;
            }
        }

        // Save positions when 'R' is pressed and saveOnReset is true
        if (saveOnReset && Input.GetKeyDown(KeyCode.R))
        {
            SavePositions();
            SaveTrialTimes();
        }
    }

    private void SavePositions()
    {
        // Write the timestamp in the first column
        long timestamp = DateTimeOffset.Now.ToUnixTimeMilliseconds();
        fileWriter.Write(timestamp.ToString());

        // Write the positions for each object in the corresponding column
        for (int i = 0; i < objectsToSave.Length; i++)
        {
            GameObject obj = objectsToSave[i];
            Vector3 position = obj.transform.position;

            fileWriter.Write($"\t{position.x},{position.y},{position.z}"); // Use tab as a separator for columns
        }
        fileWriter.Write($"\t{_camera.position.x},{_camera.position.y},{_camera.position.z}"); // Save camera position
        fileWriter.Write($"\t{_camera.rotation.x},{_camera.rotation.y},{_camera.rotation.z}"); // Save camera rotation
        fileWriter.Write($"\t{_hand.position.x},{_hand.position.y},{_hand.position.z}"); // Save hand position
        fileWriter.Write($"\t GazeTracking \t{_grab.grab}"); // Save grab status
        fileWriter.WriteLine(); // Move to the next line after writing the positions
    }

    private void SaveTrialTimes()
    {
        // Save trial times to the file and reset the PlayerPrefs value
        fileWriter.Write(PlayerPrefs.GetString("Trialtimes"));
        PlayerPrefs.SetString("Trialtimes", "");
    }

    private void OnApplicationQuit()
    {
        // Close the file when the application is quitting or the scene is stopped
        if (fileWriter != null)
        {
            fileWriter.Close();
        }
    }

    // Callback when the timer starts
    private void OnTimerStart()
    {
        PlayerPrefs.SetInt("trialtime", 1);

        // Generate a new filename based on the current time and start recording positions
        string fileName = PlayerPrefs.GetString("Mode") + "_" + System.DateTime.Now.ToString("yyyy_MM_dd_HH_mm_ss") + ".txt";
        string filePath = Path.Combine(SavePath, fileName);

        // Create the directory if it doesn't exist
        Directory.CreateDirectory(SavePath);

        // Open the file for writing
        fileWriter = new StreamWriter(filePath);

        // Write the header row with object names
        fileWriter.Write("Timestamp"); // Adding the timestamp column header

        for (int i = 0; i < objectsToSave.Length; i++)
        {
            GameObject obj = objectsToSave[i];
            fileWriter.Write("\t" + obj.name); // Use tab as a separator for columns
        }
        fileWriter.Write("\tCamPosition");
        fileWriter.Write("\tCamRotation");
        fileWriter.Write("\tHand");
        fileWriter.Write("\tGaze");
        fileWriter.Write("\tGrab");
        fileWriter.WriteLine(); // Move to the next line after writing the header row

        savePositions = true; // Start saving positions
    }

    // Callback when the timer is reset
    private void OnTimerReset()
    {
        // Close the file when the timer is reset if saveOnReset is true
        if (saveOnReset && fileWriter != null)
        {
            fileWriter.Close();
        }

        savePositions = false; // Stop saving positions
    }

    // Callback when the timer completes (hits 0)
    private void OnTimerComplete()
    {
        // Close the file when the timer completes (hits 0)
        if (fileWriter != null)
        {
            SaveTrialTimes();
            fileWriter.Close();
        }

        savePositions = false; // Stop saving positions
    }
}
