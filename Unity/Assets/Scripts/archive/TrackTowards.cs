using UnityEngine;

public class TrackTowards : MonoBehaviour
{
    public GameObject target;
    //public GameObject allObjects;
    private GameObject allObjects;
    public bool onlyOneGrabAtATime = true;

    private GameObject originalParent;
    private GameObject newTemporaryParent;
    private bool atActivation = false;

    private void ChangeParent()
    {
        if (onlyOneGrabAtATime)
        {
            // get the old parent object
            // Store the original parent of the old parent
            GameObject oldParent = transform.parent.gameObject;
            originalParent = oldParent.transform.parent.gameObject;

            // set up temporary parent
            GameObject temporaryParent = transform.Find("R_Wrist/R_Palm").gameObject;

            // Create a new game object as a copy of the temporary parent
            newTemporaryParent = Instantiate(temporaryParent);

            // Set the new temporary parent as a child of the target
            newTemporaryParent.transform.SetParent(target.transform, false);

            // Set the old parent and its children as children of the new temporary parent
            oldParent.transform.SetParent(newTemporaryParent.transform, true);

            // get the all object parent
            originalParent.transform.SetParent(allObjects.transform, false);
        }
        
    }

    private void Start()
    {
        allObjects = transform.parent.parent.gameObject;
    }

    void Update()
    {
        if (!atActivation)
        {
            // Your one-time action here
            ChangeParent();
            atActivation = true;
        }
    }

    private void OnDisable()
    {
        // Reset the flag when the script is disabled
        atActivation = false;
    }
}
