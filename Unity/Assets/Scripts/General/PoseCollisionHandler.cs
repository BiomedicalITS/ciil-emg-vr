//using Mono.Reflection;
using System.Collections.Generic;
using UnityEngine;

public class PoseCollisionHandler : MonoBehaviour
{
    public TestVsFreedom testVsFreedomScript; // Reference to the TestVsFreedom script
    public List<Transform> tables;
    public TMPro.TextMeshProUGUI instructions;

    // We need to track if the pose object is touching both tables consecutively
    private bool touchingFirstTable = false;
    private bool touchingSecondTable = false;

    public string trialtimes = "";
    private Vector3 previousPosition;
    private void Start()
    {
        PlayerPrefs.SetFloat("trialtime", 0f);
        previousPosition = transform.position;
    }
    private void Update()
    {

        if (testVsFreedomScript.Test)
        {
            // Loop through all child objects of the parent transform
            foreach (Transform child in transform)
            {
                GrabSetUp2 grabSetUp2 = child.GetComponent<GrabSetUp2>();
                if (grabSetUp2 != null && grabSetUp2.Test)
                {
                    // Check for collisions with tables in the Update method for the active pose
                    Transform collidingTable = IsPoseCollidingWithTables(transform);
                    if (collidingTable != null)
                    {
                        testVsFreedomScript.SetCollisionStatus(true);
                    }
                }
            }
        }
        previousPosition = transform.position;
    }

    private Transform IsPoseCollidingWithTables(Transform poseObject)
    {

        OnlyOneGrabAtATime onlyOneGrabAtATime = FindObjectOfType<OnlyOneGrabAtATime>();
        bool grabValue = onlyOneGrabAtATime.grab;

        BoxCollider poseCollider = poseObject.GetComponent<BoxCollider>();
        if (poseCollider != null)
        {
            foreach (Transform table in tables)
            {
                BoxCollider tableCollider = table.GetComponent<BoxCollider>();
                if (tableCollider != null && poseCollider.bounds.Intersects(tableCollider.bounds))
                {
                    if (table == tables[0] && Vector3.Distance(transform.position, previousPosition) < 0.00001f)
                    {
                        touchingFirstTable = true;
                        instructions.text = poseObject.name + " is on " + table.name + ". Move it to " + tables[1].name + ".";
                    }
                    else if (table == tables[1] && Vector3.Distance(transform.position, previousPosition) < 0.00001f)
                    {
                        touchingSecondTable = true;
                        instructions.text = poseObject.name + " is on " + table.name + ". Move it to " + tables[0].name + ".";
                    }
                }
            }
        }

        // Check if the pose object is touching both tables consecutively
        if (touchingFirstTable && touchingSecondTable && !grabValue && Vector3.Distance(transform.position, previousPosition) < 0.00001f)
        {
            float trialtime = onlyOneGrabAtATime.trialtime;

            PlayerPrefs.SetString("Trialtimes", PlayerPrefs.GetString("Trialtimes") + trialtime.ToString() + " ");
            onlyOneGrabAtATime.trialtime = 0f;

            CustomGravity customGravity = transform.GetComponent<CustomGravity>();
            customGravity.resetPosition = transform.position;
            customGravity.CheckTest = false;

            touchingFirstTable = false;
            touchingSecondTable = false;
            // The pose is colliding with both tables consecutively
            return tables[0]; // Return any table since we know it's colliding with both
        }

        return null;
    }
    
}
