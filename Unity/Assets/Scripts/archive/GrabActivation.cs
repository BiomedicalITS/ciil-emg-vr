using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;

public class GrabActivation : MonoBehaviour
{

    public GameObject poseDetection;
    public GameObject handJointPoint;
    public GameObject interactable;
    public GameObject PhantomPose;
    public float distanceForPhantomPose;
    public float distanceForGrab;
    public bool changeColours;

    // Calculates the distance between two points using Pythagorean theorem
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

    // Update is called once per frame
    void Update()
    {
        // Get the DistanceCalculator component attached to a GameObject
        GrabActivation calculator = GetComponent<GrabActivation>();

        // Calculate the distance between the transforms of the two game objects
        float distance = calculator.CalculateDistance(handJointPoint.transform, interactable.transform);

        Renderer renderer = interactable.GetComponent<Renderer>();
        //XRGrabInteractable grabInteractable = interactable.GetComponent<XRGrabInteractable>();
        
        if (poseDetection.activeSelf)
        {
            if (changeColours) {renderer.material.color = Color.blue;}

            if (distance <= distanceForPhantomPose)
            {
                // Activate phantom pose
                PhantomPose.SetActive(true);

                if (distance <= distanceForGrab)
                {
                    if (changeColours) {renderer.material.color = Color.green;}

                    //grabInteractable.enabled = true;
                    PhantomPose.SetActive(false);
                }
            }
            else
            {
                if (changeColours) {renderer.material.color = Color.blue;}

                PhantomPose.SetActive(false);
                //grabInteractable.enabled = false;
            }
        }
        else
        {
            if (changeColours) {renderer.material.color = Color.red;}

            //grabInteractable.enabled = false;
        }
    }
}
