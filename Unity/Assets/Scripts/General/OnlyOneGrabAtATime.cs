using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class OnlyOneGrabAtATime : MonoBehaviour
{
    public List<GameObject> AllPoses; // List of all pose GameObjects
    public GameObject GrabStatusObject; // Object to indicate grab status
    private VTFeedback _tactor; // Reference to VTFeedback component
    private EMGSetUp _emg; // Reference to EMGSetUp component
    public Color targetColor = Color.green; // Desired color when grabbed
    public float trialtime = 0f; // Variable to track trial time

    private Renderer objectRenderer; // Renderer of the GrabStatusObject
    private Color originalColor = Color.red; // Original color of the GrabStatusObject
    public bool grab = false; // Indicates if a grab is active

    // Check if any object is being grabbed and update the grab status
    private void CheckGrab()
    {
        bool shouldEnableOnlyOneGrabAtATime = true; // Track if only one grab should be enabled

        grab = false; // Reset grab to false at the start of each check

        foreach (GameObject obj in AllPoses)
        {
            TrackTowards component = obj.GetComponent<TrackTowards>();

            if (component != null && component.enabled)
            {
                grab = true;
                shouldEnableOnlyOneGrabAtATime = false; // At least one component has grab = true
                SetOnlyOneGrabAtATimeForAllObjects(obj);
                break;
            }
        }

        if (shouldEnableOnlyOneGrabAtATime)
        {
            EnableOnlyOneGrabAtATimeForAllObjects();
        }
    }

    // Disable the grab functionality for all objects except the current one
    private void SetOnlyOneGrabAtATimeForAllObjects(GameObject currentObject)
    {
        foreach (GameObject obj in AllPoses)
        {
            if (obj != currentObject)
            {
                TrackTowards component = obj.GetComponent<TrackTowards>();
                if (component != null)
                {
                    component.onlyOneGrabAtATime = false;
                }
            }
        }
    }

    // Enable the grab functionality for all objects
    private void EnableOnlyOneGrabAtATimeForAllObjects()
    {
        foreach (GameObject obj in AllPoses)
        {
            TrackTowards component = obj.GetComponent<TrackTowards>();
            if (component != null)
            {
                component.onlyOneGrabAtATime = true;
            }
        }
    }

    // Start is called before the first frame update
    private void Start()
    {
        // Initialize object references and settings
        objectRenderer = GrabStatusObject.GetComponent<Renderer>();
        originalColor = objectRenderer.material.color;
        _emg = FindObjectOfType<EMGSetUp>();
        if (_emg.VT)
        {
            _tactor = FindObjectOfType<VTFeedback>();
            _tactor.Write(false);
        }
    }

    // Update is called once per frame
    void Update()
    {
        trialtime += Time.deltaTime; // Increment trial time
        CheckGrab(); // Check grab status
        int FeedbackMode = PlayerPrefs.GetInt("FeedbackMode");

        // Handle feedback modes based on the FeedbackMode value
        if (FeedbackMode == 0)
        {
            GrabStatusObject.SetActive(false);
        }
        if (FeedbackMode == 1) // Visual feedback
        {
            GrabStatusObject.SetActive(true);
            if (grab)
            {
                objectRenderer.material.color = targetColor;
            }
            else
            {
                objectRenderer.material.color = originalColor;
            }
        }
        if (FeedbackMode == 2 && _emg.VT) // Vibrotactile feedback
        {
            GrabStatusObject.SetActive(false);
            if (grab)
            {
                _tactor.Write(true);
            }
            else
            {
                _tactor.Write(false);
            }
        }
    }
}
