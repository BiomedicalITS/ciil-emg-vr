using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Hands;

[System.Serializable]
public class GestureActivation
{
    public string gestureName;
    public bool isActive;
    public List<XRHandTrackingEvents> handTrackingEvents;
    public List<SkinnedMeshRenderer> skinnedMeshRenderers;

    public GestureActivation()
    {
        handTrackingEvents = new List<XRHandTrackingEvents>();
        skinnedMeshRenderers = new List<SkinnedMeshRenderer>();
    }
}

public class HandGestureActivation : MonoBehaviour
{
    public EMGRawReader emgRawReader; // Public parameter to receive EMGRawReader
    public List<GestureActivation> gestureActivationStatus;

    private static HandGestureActivation instance;

    void Awake()
    {
        if (instance == null)
        {
            instance = this;
            DontDestroyOnLoad(this.gameObject);
        }
        else
        {
            Destroy(this.gameObject);
        }
    }

    void Start()
    {
        emgRawReader = FindObjectOfType<EMGRawReader>();
    }

    void Update()
    {
        int prediction = emgRawReader.EMG_prediction;
        for (int idx = 0; idx < gestureActivationStatus.Count; idx++)
        {
            bool shouldActivate = (idx == prediction);
            if (gestureActivationStatus[idx].isActive != shouldActivate)
            {
                gestureActivationStatus[idx].isActive = shouldActivate;
                // Update the activation status of XRHandTrackingEvents and SkinnedMeshRenderers
                UpdateActivationStatus(gestureActivationStatus[idx]);
            }
        }
    }

    public void Activate(string gestureName)
    {
        // Check if the gesture is already in the list
        GestureActivation gestureActivation = gestureActivationStatus.Find(g => g.gestureName == gestureName);
        if (gestureActivation == null)
        {
            // If not, add it with a default value of false
            gestureActivation = new GestureActivation { gestureName = gestureName, isActive = false };
            gestureActivationStatus.Add(gestureActivation);
        }

        // Find the child object by name
        Transform gestureObjectTransform = transform.Find(gestureName);
        if (gestureObjectTransform == null)
        {
            Debug.LogError($"Gesture object '{gestureName}' not found");
            return;
        }

        GameObject gestureObject = gestureObjectTransform.gameObject;

        // Find and add all XRHandTrackingEvents components in children
        XRHandTrackingEvents[] handTrackingEvents = gestureObject.GetComponentsInChildren<XRHandTrackingEvents>();
        if (handTrackingEvents != null)
        {
            gestureActivation.handTrackingEvents.AddRange(handTrackingEvents);
        }

        // Find and add all SkinnedMeshRenderer components in RightHand children
        Transform rightHandTransform = gestureObjectTransform.Find("RightHand");
        if (rightHandTransform != null)
        {
            SkinnedMeshRenderer[] skinnedMeshRenderers = rightHandTransform.GetComponentsInChildren<SkinnedMeshRenderer>();
            if (skinnedMeshRenderers != null)
            {
                gestureActivation.skinnedMeshRenderers.AddRange(skinnedMeshRenderers);
            }
        }
    }

    private void UpdateActivationStatus(GestureActivation gestureActivation)
    {
        // Update XRHandTrackingEvents
        foreach (var handTrackingEvent in gestureActivation.handTrackingEvents)
        {
            handTrackingEvent.enabled = gestureActivation.isActive;
        }

        // Update SkinnedMeshRenderers
        foreach (var skinnedMeshRenderer in gestureActivation.skinnedMeshRenderers)
        {
            skinnedMeshRenderer.enabled = gestureActivation.isActive;
        }
    }
}
