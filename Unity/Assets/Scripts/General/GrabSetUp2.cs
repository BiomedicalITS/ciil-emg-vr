using System.Collections.Generic;
using UnityEngine;

public class GrabSetUp2 : MonoBehaviour
{
    public GameObject AllowedPose; // The pose that is allowed for grabbing
    public GameObject Target; // The target object for grabbing

    public bool Test; // Indicates if the object is in test mode
    public bool Freedom; // Indicates if the object is in freedom mode

    private float DistanceForPhantomPosesMin = 0.1f; // Minimum distance for phantom poses
    private float DistanceForPhantomPosesMax = 0.5f; // Maximum distance for phantom poses

    private float DistanceForGrab = 0.1f; // Distance threshold for grabbing

    private bool follow = false; // Indicates if the object should follow
    private bool testfollow = false; // Indicates if the test object should follow

    private string last_timestamp = "0";

    private bool flag = true; // Flag to check if the test mode should start
    private EMGRawReader _emgRawReader; // Reference to EMGRawReader component
    private EMGSetUp _emg; // Reference to EMGSetUp component

    private List<GameObject> AllowedPosePositions = new List<GameObject>(); // List of allowed pose positions
    private List<GameObject> AllowedPhantomPosePositions = new List<GameObject>(); // List of allowed phantom pose positions

    private MeshRenderer childCRenderer; // Renderer for child components

    // Get the context for the current object grab classes
    private string GetContext(string obj_grab_classes)
    {
        string[] pose_names = { "Ne", "H1", "H2", "H3", "H4", "T1", "T2", "T3", "T4" };
        string pose_name = pose_names[_emgRawReader.EMG_prediction];

        string context;
        if (obj_grab_classes.Contains(pose_name))
        {
            context = "P ";
        }
        else
        {
            context = "N ";
        }
        return context;
    }

    // Get the contraction string for the current object
    private string GetContraction()
    {
        string contraction = " ";
        Transform parent = transform.parent;

        if (parent != null)
        {
            int childCount = parent.childCount;
            for (int i = 0; i < childCount; i++)
            {
                Transform child = parent.GetChild(i);
                string firstTwoLetters = child.name[..Mathf.Min(2, child.name.Length)];
                contraction += firstTwoLetters;
                if (i < childCount - 1)
                {
                    contraction += " ";
                }
            }
        }
        return contraction;
    }

    // Calculate the distance between two transforms
    public float CalculateDistance(Transform objectA, Transform objectB)
    {
        Vector3 positionA = objectA.position;
        Vector3 positionB = objectB.position;

        float deltaX = positionB.x - positionA.x;
        float deltaY = positionB.y - positionA.y;
        float deltaZ = positionB.z - positionA.z;

        float distanceSquared = (deltaX * deltaX) + (deltaY * deltaY) + (deltaZ * deltaZ);
        float distance = Mathf.Sqrt(distanceSquared);

        return distance;
    }

    // Handle grab activation in freedom mode
    public void GrabActivationFreedom()
    {
        float distance = CalculateDistance(Target.transform, transform.parent.transform);

        SkinnedMeshRenderer GhostAppear = transform.GetComponentInChildren<SkinnedMeshRenderer>();

        if (distance >= DistanceForPhantomPosesMin && distance <= DistanceForPhantomPosesMax)
        {
            GhostAppear.enabled = true; // Activate phantom pose
        }
        if (distance < DistanceForPhantomPosesMin && !follow)
        {
            GhostAppear.enabled = false; // Deactivate phantom pose
        }
        if (distance >= DistanceForPhantomPosesMax)
        {
            GhostAppear.enabled = false; // Deactivate phantom pose
        }
        TrackTowards component = GetComponent<TrackTowards>();
        if (!component.onlyOneGrabAtATime)
        {
            GhostAppear.enabled = false; // Deactivate phantom pose
        }
    }

    // Handle grab activation in test mode
    public void GrabActivationTest()
    {
        float distance = CalculateDistance(Target.transform, transform.parent.transform);

        Transform parent = transform.parent;
        testfollow = false;
        foreach (Transform child in parent)
        {
            TrackTowards component = child.GetComponent<TrackTowards>();
            testfollow = testfollow || component.enabled;
        }
        foreach (Transform child in parent)
        {
            TrackTowards component = child.GetComponent<TrackTowards>();
            child.transform.Find("RightHand").GetComponent<SkinnedMeshRenderer>().enabled = (testfollow && component.enabled) || (!testfollow && !component.enabled);
        }

        if (distance < DistanceForPhantomPosesMax && _emg.EMG)
        {
            string timestamp = _emgRawReader.timestamp;

            // Don't bother if the timestamp hasn't changed
            if (timestamp == last_timestamp)
                return;
            last_timestamp = timestamp;

            // Don't consider idle class for context
            if (_emgRawReader.EMG_prediction == 0 || _emgRawReader.EMG_prediction == 7)
                return;

            string contraction = GetContraction();
            string context = GetContext(contraction);
            string message = context + timestamp + contraction;
            Debug.Log(message);
            _emgRawReader.Write(message);
        }
    }

    // Initialize the component
    void Start()
    {
        _emgRawReader = FindObjectOfType<EMGRawReader>();
        _emg = FindObjectOfType<EMGSetUp>();

        // Initialize the allowed pose positions
        GameObject palm = AllowedPose.transform.Find("R_Wrist/R_Palm").gameObject;
        AllowedPosePositions.Add(palm);

        // Initialize the allowed phantom pose positions
        GameObject palmG = transform.Find("R_Wrist/R_Palm").gameObject;
        AllowedPhantomPosePositions.Add(palmG);
    }

    // Handle the grab logic
    public void GrabHandeller()
    {
        float meanDistance = CalculateDistance(AllowedPosePositions[0].transform, AllowedPhantomPosePositions[0].transform);

        if (meanDistance < DistanceForGrab)
        {
            follow = true;
        }
        SkinnedMeshRenderer GhostAppear = GetComponentInChildren<SkinnedMeshRenderer>();
        childCRenderer = AllowedPose.transform.Find("RightHand/Sphere").GetComponent<MeshRenderer>();

        if (follow)
        {
            GetComponent<TrackTowards>().enabled = true;
            GhostAppear.enabled = true;
            childCRenderer.enabled = false;
            follow = false;
        }

        if (!AllowedPose.activeSelf)
        {
            follow = false;
            childCRenderer.enabled = true;
            GetComponent<TrackTowards>().enabled = false;
            AllowedPose.transform.position = new Vector3(0, 0, 0);
        }
    }

    // Update is called once per frame
    void Update()
    {
        CustomGravity customGravity = transform.parent.GetComponent<CustomGravity>();
        if (Freedom)
        {
            GrabActivationFreedom();
        }
        if (Test)
        {
            if (flag)
            {
                customGravity.CheckTest = true;
                flag = false;
            }
            GrabActivationTest();
        }
        if (!Freedom && !Test && !customGravity.CheckTest)
        {
            SkinnedMeshRenderer GhostAppear = GetComponentInChildren<SkinnedMeshRenderer>();
            GhostAppear.enabled = false;
        }
        GrabHandeller();
    }
}
