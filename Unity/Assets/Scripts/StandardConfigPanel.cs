using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class StandardConfigPanel : MonoBehaviour
{
    public SGTManager sgtManager; // Reference to the SGTManager script
    public int defaultNumReps = 10;
    public int defaultTimePerRep = 2;
    public int defaultTimeBetRep = 1;
    public string defaultOutputFolderText = "DefaultOutputFolder"; // Replace with your default text
    //public string defaultInputDirectory = "Assets/Scripts/SGT/DefaultInput"; // Replace with your actual directory

    // Update is called once per frame
    void Update()
    {
        // Check if Ctrl and Enter are pressed simultaneously
        if (Input.GetKey(KeyCode.LeftControl) && Input.GetKeyDown(KeyCode.Return))
        {
            // Check if the SGTManager reference is not null
            if (sgtManager != null)
            {
                // Assign predetermined values to the sliders using properties
                sgtManager._numRepsSlider.value = defaultNumReps;
                sgtManager._timePerRepSlider.value = defaultTimePerRep;
                sgtManager._timeBetRepSlider.value = defaultTimeBetRep;

                // Check if there are images in the defaultInputDirectory
               // string[] imagePaths = System.IO.Directory.GetFiles(defaultInputDirectory, "*.png")
               //                     .Concat(System.IO.Directory.GetFiles(defaultInputDirectory, "*.jpg"))
               //                     .Concat(System.IO.Directory.GetFiles(defaultInputDirectory, "*.svg"))
               //                     .ToArray();

                //if (imagePaths.Length > 0)
                //{
                    // Call SelectInputClicked
                sgtManager.SelectInputClicked();
                //}
                //else
                //{
                //    Debug.LogWarning("No images found in the specified directory.");
                //}

                // Call SelectOutputClicked
                sgtManager.SelectOutputClicked();
            }
            else
            {
                Debug.LogError("SGTManager reference is null. Please assign it in the Inspector.");
            }
        }
    }
}
