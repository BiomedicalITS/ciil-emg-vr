using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PoseChanger : MonoBehaviour
{
    // Reference to the GameObject that holds all objects
    private GameObject allObjects;

    // Lists to hold the state and references of the poses
    private List<bool> poseState;
    private List<GameObject> poseList;

    // References to external scripts
    private EMGRawReader EMG_classifier;
    private EMGSetUp _emg;

    // Previous prediction value
    private float previousVal = 100;

    // Method to change the state of a specific pose
    public void ChangeState(int index)
    {
        bool currentState = poseState[index];
        poseState[index] = !currentState;

        // Deactivate all other poses
        for (int i = 0; i < poseState.Count; i++)
        {
            if (i != index)
            {
                poseState[i] = false;
                poseList[i].SetActive(false);
            }
        }
    }

    // Method to destroy child objects
    private void DestroyChild(List<GameObject> poseList)
    {
        foreach (GameObject pose in poseList)
        {
            foreach (Transform child in pose.transform.Find("R_Wrist/R_Palm"))
            {
                if (child.name.Contains("Clone"))
                {
                    Transform firstChild = child.GetChild(0);
                    GameObject childObject = firstChild.gameObject;

                    // Disable the TrackTowards component if it exists
                    foreach (Transform ghost in childObject.transform)
                    {
                        TrackTowards trackedComponent = ghost.GetComponent<TrackTowards>();
                        if (trackedComponent != null)
                        {
                            trackedComponent.enabled = false;
                        }
                    }

                    // Move the child object to the allObjects parent
                    firstChild.transform.SetParent(allObjects.transform);
                    Destroy(child.gameObject); // Destroy the original child game object
                }
            }
        }
    }

    // Start is called before the first frame update
    private void Start()
    {
        // Initialize references to objects and components
        allObjects = transform.parent.Find("NewObjects v2").gameObject;
        EMG_classifier = FindObjectOfType<EMGRawReader>();
        _emg = FindObjectOfType<EMGSetUp>();

        // Initiate poseList with all child objects of the "Right" transform
        poseList = new List<GameObject>();
        Transform rightTransform = transform.Find("Right");
        foreach (Transform child in rightTransform)
        {
            poseList.Add(child.gameObject);
        }

        // Initiate poseState with the same size as poseList
        poseState = new List<bool>();
        for (int i = 0; i < poseList.Count; i++)
        {
            poseState.Add(i == 0); // Set the first pose active and others inactive
        }
    }

    // Update is called once per frame
    void Update()
    {
        // Get EMG accuracy and prediction values
        float velocity = EMG_classifier.EMG_velocity;
        int prediction = EMG_classifier.EMG_prediction;
        PoseChanger poseStateManager = GetComponent<PoseChanger>();

        // Change pose based on EMG prediction if the value has changed
        if (_emg.EMG && previousVal != prediction)
        {
            previousVal = prediction;

            // Handle pose changes based on prediction value
            if (prediction == 0)
            {
                DestroyChild(poseList);
                poseList[0].SetActive(!poseState[0]);
                poseStateManager.ChangeState(0);
            }
            else if (prediction == 1)
            {
                DestroyChild(poseList);
                poseList[1].SetActive(!poseState[1]);
                poseStateManager.ChangeState(1);
            }
            else if (prediction == 2)
            {
                DestroyChild(poseList);
                poseList[2].SetActive(!poseState[2]);
                poseStateManager.ChangeState(2);
            }
            else if (prediction == 3)
            {
                DestroyChild(poseList);
                poseList[3].SetActive(!poseState[3]);
                poseStateManager.ChangeState(3);
            }
            else if (prediction == 4)
            {
                DestroyChild(poseList);
                poseList[4].SetActive(!poseState[4]);
                poseStateManager.ChangeState(4);
            }
            else if (prediction == 5)
            {
                DestroyChild(poseList);
                poseList[5].SetActive(!poseState[5]);
                poseStateManager.ChangeState(5);
            }
            else if (prediction == 6)
            {
                DestroyChild(poseList);
                poseList[6].SetActive(!poseState[6]);
                poseStateManager.ChangeState(6);
            }
            else if (prediction == 7)
            {
                DestroyChild(poseList);
                poseList[7].SetActive(!poseState[7]);
                poseStateManager.ChangeState(7);
            }
            else if (prediction == 8)
            {
                DestroyChild(poseList);
                poseList[8].SetActive(!poseState[8]);
                poseStateManager.ChangeState(8);
            }
        }

        // Change pose based on numeric key input
        if (Input.GetKeyDown(KeyCode.Keypad1))
        {
            DestroyChild(poseList);
            poseList[0].SetActive(!poseState[0]);
            poseStateManager.ChangeState(0);
        }
        else if (Input.GetKeyDown(KeyCode.Keypad2))
        {
            DestroyChild(poseList);
            poseList[1].SetActive(!poseState[1]);
            poseStateManager.ChangeState(1);
        }
        else if (Input.GetKeyDown(KeyCode.Keypad3))
        {
            DestroyChild(poseList);
            poseList[2].SetActive(!poseState[2]);
            poseStateManager.ChangeState(2);
        }
        else if (Input.GetKeyDown(KeyCode.Keypad4))
        {
            DestroyChild(poseList);
            poseList[3].SetActive(!poseState[3]);
            poseStateManager.ChangeState(3);
        }
        else if (Input.GetKeyDown(KeyCode.Keypad5))
        {
            DestroyChild(poseList);
            poseList[4].SetActive(!poseState[4]);
            poseStateManager.ChangeState(4);
        }
        else if (Input.GetKeyDown(KeyCode.Keypad6))
        {
            DestroyChild(poseList);
            poseList[5].SetActive(!poseState[5]);
            poseStateManager.ChangeState(5);
        }
        else if (Input.GetKeyDown(KeyCode.Keypad7))
        {
            DestroyChild(poseList);
            poseList[6].SetActive(!poseState[6]);
            poseStateManager.ChangeState(6);
        }
        else if (Input.GetKeyDown(KeyCode.Keypad8))
        {
            DestroyChild(poseList);
            poseList[7].SetActive(!poseState[7]);
            poseStateManager.ChangeState(7);
        }
        else if (Input.GetKeyDown(KeyCode.Keypad9))
        {
            DestroyChild(poseList);
            poseList[8].SetActive(!poseState[8]);
            poseStateManager.ChangeState(8);
        }
    }
}
