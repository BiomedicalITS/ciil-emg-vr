using UnityEngine;
using System.Collections.Generic;

public class CustomGravity : MonoBehaviour
{
    public float gravityForce = 0.2f; // Adjust this value to control the strength of gravity
    public List<Transform> tables; // List of tables, drag and drop them in the Inspector
    public Vector3 resetPosition;

    public bool CheckTest = false;

    private string tableName;
    private void Start()
    {
        resetPosition = transform.position;
    }

    private void Update()
    {
        bool collidesWithTable = false;

        // Check if the object collides with any of the tables
        foreach (Transform table in tables)
        {
            if (GetComponent<BoxCollider>().bounds.Intersects(table.GetComponent<BoxCollider>().bounds))
            {
                collidesWithTable = true;
                tableName = table.name;
                break;
            }
        }

        bool hasChildWithTrackTowardsActive = false;

        // Check if any of the object's children have the TrackTowards component active
        foreach (Transform child in transform)
        {
            TrackTowards trackTowardsComponent = child.GetComponent<TrackTowards>();
            if (trackTowardsComponent != null && trackTowardsComponent.isActiveAndEnabled)
            {
                hasChildWithTrackTowardsActive = true;
                break;
            }
        }
        // Set gravityForce to 0 if any child has the TrackTowards component active
        if (hasChildWithTrackTowardsActive)
        {
            gravityForce = 0f;
        }
        // Set gravityForce to 0 if the object collides with any table
        if (collidesWithTable)
        {
            //Debug.Log($"{transform.name} + is on the {tableName}");
            gravityForce = 0f;
        }
        if (!hasChildWithTrackTowardsActive && !collidesWithTable)
        {
            gravityForce = 0.2f; // Reset gravityForce to its original value
        }

        // Apply gravity force to the object
        Vector3 gravityVector = new Vector3(0f, -gravityForce, 0f);
        transform.position += gravityVector * Time.deltaTime;

        // Reset the position if the height is lower than the y value of the referenceObject
        if (transform.position.y < GetLowestTableHeight())
        {
            transform.position = resetPosition;
        }
    }

    // Helper function to get the lowest height among all the tables
    private float GetLowestTableHeight()
    {
        float lowestHeight = Mathf.Infinity;
        foreach (Transform table in tables)
        {
            if (table.position.y < lowestHeight)
            {
                lowestHeight = table.position.y;
            }
        }
        return (float)(lowestHeight - 0.5);
    }
}