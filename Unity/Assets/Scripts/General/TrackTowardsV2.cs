using UnityEngine;

public class TrackTowardsV2 : MonoBehaviour
{

    public GameObject target; //the enemy's target

    void Update()
    {
        //move towards the player
        transform.SetPositionAndRotation(target.transform.position, target.transform.rotation);
    }

}