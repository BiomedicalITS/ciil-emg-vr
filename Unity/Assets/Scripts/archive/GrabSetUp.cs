using System.Collections.Generic;
using UnityEngine;

public class GrabSetUp : MonoBehaviour
{
    public List<GameObject> AllowedPoses;
    public List<GameObject> AllowedPhantomPoses;
    
    public GameObject TargetAnchor;
    public GameObject TargetRotationAnchor;
    public GameObject Interactable;

    public float DistanceForPhantomPoses;
    public float DistanceForGrab;
    public bool ChangeColors;

    public Vector3 deltaPosition;
    public Vector3 deltaRotation;

    private List<List<GameObject>> AllowedPosesPositions = new List<List<GameObject>>();
    private List<List<GameObject>> AllowedPhantomPosesPositions = new List<List<GameObject>>();

    private bool follow = false;
    Quaternion rotgoal;
    Vector3 direction;
    public float turnspeed = .01f;

    public float CalculateDistance(Transform objectA, Transform objectB)
    {
        // Get the positions of the two objects
        Vector3 positionA = objectA.position;
        Vector3 positionB = objectB.position;

        // Calculate the differences in coordinates
        float deltaX = positionB.x - positionA.x;
        float deltaY = positionB.y - positionA.y;
        float deltaZ = positionB.z - positionA.z;

        // Calculate the squared distance using Pythagorean theorem
        float distanceSquared = (deltaX * deltaX) + (deltaY * deltaY) + (deltaZ * deltaZ);

        // Calculate the actual distance by taking the square root of the squared distance
        float distance = Mathf.Sqrt(distanceSquared);

        return distance;
    }

    public int IsAPoseActive()
    {
        foreach (GameObject pose in AllowedPoses)
        {
            if (pose.activeSelf)
            {
                return AllowedPoses.IndexOf(pose);
            }
        }
        return -1;
    }

    public int GrabActivation()
    {
        // Get the DistanceCalculator component attached to a GameObject
        GrabSetUp calculator = GetComponent<GrabSetUp>();

        // Calculate the distance between the transforms of the two game objects
        float distance = calculator.CalculateDistance(TargetAnchor.transform, Interactable.transform);

        Renderer renderer = Interactable.GetComponent<Renderer>();

        int ActivePoseIndex = IsAPoseActive();
        if (ActivePoseIndex > -1)
        {
            if (ChangeColors) { renderer.material.color = Color.blue; }

            if (distance <= DistanceForPhantomPoses)
            {
                // Activate phantom pose
                AllowedPhantomPoses[ActivePoseIndex].SetActive(!follow);

                if (distance <= DistanceForGrab)
                {
                    if (ChangeColors) { renderer.material.color = Color.green; }
                }
            }
            else
            {
                if (ChangeColors) { renderer.material.color = Color.blue; }
                AllowedPhantomPoses[ActivePoseIndex].SetActive(false);
            }
        }
        else
        {
            if (ChangeColors) { renderer.material.color = Color.red; }
        }
        return ActivePoseIndex;
    }

    public void GrabHandeller(int ActivePoseIndex)
    {
        float sum = 0;
        if (ActivePoseIndex > -1)
        {
            for (int i = 0; i < AllowedPosesPositions[ActivePoseIndex].Count; i++)
            {
                sum += CalculateDistance(AllowedPosesPositions[ActivePoseIndex][i].transform, AllowedPhantomPosesPositions[ActivePoseIndex][i].transform);
            }
        }


        if (sum < DistanceForGrab && ActivePoseIndex > -1)
        {
            follow = true;
        }

        if (follow)
        {
            transform.rotation = TargetRotationAnchor.transform.rotation * Quaternion.Euler(deltaRotation.x, deltaRotation.y, deltaRotation.z);

            transform.position = TargetAnchor.transform.position + deltaPosition ;

        }
        if (ActivePoseIndex == -1)
        {
            follow = false;
            transform.position = transform.position;
        }
    }
    void Start()
    {
        AllowedPosesPositions = new List<List<GameObject>>();

        for (int poseNumber = 0; poseNumber < AllowedPoses.Count; poseNumber++) //iterate over all Allowed poses
        {
            // gotta make the inner list so we can add to that
            AllowedPosesPositions.Add(new List<GameObject>());

            //index tip
            GameObject IndexTip = AllowedPoses[poseNumber].transform.Find("R_Wrist/R_IndexMetacarpal/R_IndexProximal/R_IndexIntermediate/R_IndexDistal/R_IndexTip").gameObject;

            //middle tip
            GameObject MiddleTip = AllowedPoses[poseNumber].transform.Find("R_Wrist/R_MiddleMetacarpal/R_MiddleProximal/R_MiddleIntermediate/R_MiddleDistal/R_MiddleTip").gameObject;

            //ring tip
            GameObject RingTip = AllowedPoses[poseNumber].transform.Find("R_Wrist/R_RingMetacarpal/R_RingProximal/R_RingIntermediate/R_RingDistal/R_RingTip").gameObject;

            //thumb tip
            GameObject ThumbTip = AllowedPoses[poseNumber].transform.Find("R_Wrist/R_ThumbMetacarpal/R_ThumbProximal/R_ThumbDistal/R_ThumbTip").gameObject;

            //palm
            GameObject palm = AllowedPoses[poseNumber].transform.Find("R_Wrist/R_Palm").gameObject;

            AllowedPosesPositions[poseNumber].Add(IndexTip);
            AllowedPosesPositions[poseNumber].Add(MiddleTip);
            AllowedPosesPositions[poseNumber].Add(RingTip);
            AllowedPosesPositions[poseNumber].Add(ThumbTip);
            AllowedPosesPositions[poseNumber].Add(palm);
        }

        AllowedPhantomPosesPositions = new List<List<GameObject>>();

        for (int poseNumber = 0; poseNumber < AllowedPhantomPoses.Count; poseNumber++) //iterate over all Allowed poses
        {
            // gotta make the inner list so we can add to that
            AllowedPhantomPosesPositions.Add(new List<GameObject>());

            //index tip
            GameObject IndexTip = AllowedPhantomPoses[poseNumber].transform.Find("R_Wrist/R_IndexMetacarpal/R_IndexProximal/R_IndexIntermediate/R_IndexDistal/R_IndexTip").gameObject;

            //middle tip
            GameObject MiddleTip = AllowedPhantomPoses[poseNumber].transform.Find("R_Wrist/R_MiddleMetacarpal/R_MiddleProximal/R_MiddleIntermediate/R_MiddleDistal/R_MiddleTip").gameObject;

            //ring tip
            GameObject RingTip = AllowedPhantomPoses[poseNumber].transform.Find("R_Wrist/R_RingMetacarpal/R_RingProximal/R_RingIntermediate/R_RingDistal/R_RingTip").gameObject;

            //thumb tip
            GameObject ThumbTip = AllowedPhantomPoses[poseNumber].transform.Find("R_Wrist/R_ThumbMetacarpal/R_ThumbProximal/R_ThumbDistal/R_ThumbTip").gameObject;

            //palm
            GameObject palm = AllowedPhantomPoses[poseNumber].transform.Find("R_Wrist/R_Palm").gameObject;

            AllowedPhantomPosesPositions[poseNumber].Add(IndexTip);
            AllowedPhantomPosesPositions[poseNumber].Add(MiddleTip);
            AllowedPhantomPosesPositions[poseNumber].Add(RingTip);
            AllowedPhantomPosesPositions[poseNumber].Add(ThumbTip);
            AllowedPhantomPosesPositions[poseNumber].Add(palm);
        }
    }

    // Update is called once per frame
    void Update()
    {
        int ActivePoseIndex = GrabActivation();
        GrabHandeller(ActivePoseIndex);
    }
}