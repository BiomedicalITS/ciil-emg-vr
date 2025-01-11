using UnityEngine;
using System.Collections.Generic;
using UnityEngine.XR.OpenXR.Input;
using Unity.VRTemplate;

public class TestVsFreedom : MonoBehaviour
{
    public List<GameObject> poses;
    public bool Freedom = false;
    public TMPro.TextMeshProUGUI instructions;
    public TMPro.TextMeshProUGUI timer;
    public MeshRenderer visualIndicator;
    public bool Test = false;
    public bool TestVisual = false;
    public bool TestVibro = false;
    public bool SGT = false;
    public List<Transform> tables;

    private bool collision = false;
    private GameObject currentTestObject = null;
    private HashSet<GameObject> completedTasks = new HashSet<GameObject>();
    private GameObject RightHand;
    private GameObject RightHandPoses;
    [SerializeField] private GameObject _sgtPanel;

    private readonly int[] poses_to_do = { 0, 13, 25, 7, 44, 42 };

    // Initialize the random state and set up initial component states
    private void Start()
    {
        Random.InitState(1);
        DeactivateComponents();
        instructions.enabled = true;
        timer.enabled = false;
        visualIndicator.enabled = false;
    }

    // Update is called once per frame
    private void Update()
    {
        // Handle Escape key press
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            Test = false;
            timer.enabled = false;
            instructions.enabled = true;
            visualIndicator.enabled = false;

            RightHand = transform.parent.Find("Complete XR Origin Hands Set Up/XR Origin (XR Rig)/Camera Offset/Right Hand/Right Hand Interaction Visual/RightHand").gameObject;
            RightHand.SetActive(false);

            RightHandPoses = transform.parent.Find("Poses2 v2").gameObject;
            RightHandPoses.SetActive(true);
            DeactivateComponents();
            return;
        }

        // Handle O key press to start test mode
        if (Input.GetKeyDown(KeyCode.O))
        {
            if (!Test)
            {
                Debug.Log("Test mode enabled");
                Test = true;
                timer.enabled = true;
                visualIndicator.enabled = false;
                PlayerPrefs.SetInt("FeedbackMode", 0);
                PlayerPrefs.SetString("Mode", "OL");

                StartTestMode();
            }
        }

        // Handle V key press to start visual feedback test mode
        if (Input.GetKeyDown(KeyCode.V))
        {
            if (!Test)
            {
                Debug.Log("Visual feedback test mode enabled");
                Test = true;
                timer.enabled = true;
                instructions.enabled = true;
                visualIndicator.enabled = true;
                PlayerPrefs.SetInt("FeedbackMode", 1);
                PlayerPrefs.SetString("Mode", "TV");

                StartTestMode();
            }
        }

        // Handle F key press to start exploration mode
        if (Input.GetKeyDown(KeyCode.F))
        {
            if (!Test)
            {
                Debug.Log("Exploration mode enabled");
                timer.enabled = true;
                instructions.enabled = true;
                visualIndicator.enabled = true;

                StartFreedomMode();
            }
        }

        // Check for object collision status to change the test object
        if (collision)
        {
            collision = false;
            ChangeRandomTestObject();
        }
    }

    // Start the test mode by deactivating SGT panel and setting a random pose
    private void StartTestMode()
    {
        _sgtPanel.SetActive(false);
        GameObject randomPose = GetRandomPose();
        SetTestObject(randomPose);
    }

    // Start the freedom mode with appropriate instructions
    private void StartFreedomMode()
    {
        _sgtPanel.SetActive(false);
        Test = false;
        instructions.text = "Exploration mode. Try to grab objects with different contractions";
        SetAllPosesFreedom();
    }

    // Set the current test object and update its state
    private void SetTestObject(GameObject pose)
    {
        currentTestObject = pose;
        foreach (GameObject p in poses)
        {
            GrabSetUp2 grabSetUp2 = p.GetComponent<GrabSetUp2>();
            if (grabSetUp2 != null)
            {
                grabSetUp2.Test = p == pose;
                grabSetUp2.Freedom = false;
            }
        }
    }

    // Set all poses to freedom mode
    private void SetAllPosesFreedom()
    {
        foreach (GameObject pose in poses)
        {
            GrabSetUp2 grabSetUp2 = pose.GetComponent<GrabSetUp2>();
            if (grabSetUp2 != null)
            {
                grabSetUp2.Test = false;
                grabSetUp2.Freedom = true;
            }
        }
    }

    // Change the current test object to a new random one
    private void ChangeRandomTestObject()
    {
        completedTasks.Add(currentTestObject);

        // Find a new random test object that has not been tested yet
        GameObject newRandomPose = GetRandomPose();

        if (newRandomPose == null)
        {
            // disable everything
            Test = false;
            timer.enabled = false;
            instructions.enabled = true;
            visualIndicator.enabled = false;

            RightHand = transform.parent.Find("Complete XR Origin Hands Set Up/XR Origin (XR Rig)/Camera Offset/Right Hand/Right Hand Interaction Visual/RightHand").gameObject;
            RightHand.SetActive(false);

            RightHandPoses = transform.parent.Find("Poses2 v2").gameObject;
            RightHandPoses.SetActive(true);
            DeactivateComponents();
            return;
        }

        // Set the new test object
        SetTestObject(newRandomPose);
    }

    // Get a random pose from the list of poses
    // EDIT 2024-07-04: not random :)
    private GameObject GetRandomPose()
    {
        if (poses.Count == 0)
            return null;


        int num_done = completedTasks.Count;
        if (num_done >= poses_to_do.Length)
        {
            return null;
        }

        return poses[poses_to_do[num_done]];
    }

    // Set the collision status for the test object
    public void SetCollisionStatus(bool colliding)
    {
        collision = colliding;
    }

    // Deactivate components and reset states
    private void DeactivateComponents()
    {
        foreach (GameObject pose in poses)
        {
            GrabSetUp2 grabSetUp2 = pose.GetComponent<GrabSetUp2>();
            CustomGravity customGravity = pose.transform.parent.GetComponent<CustomGravity>();
            if (grabSetUp2 != null)
            {
                customGravity.CheckTest = false;
                grabSetUp2.Test = false;
                grabSetUp2.Freedom = false;

                instructions.text = "F: Freedom, O: Testing, V: Visual Test";
            }
        }

        Test = false;
        TestVisual = false;
        TestVibro = false;
        SGT = false;
        _sgtPanel.SetActive(false);
    }
}
