using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

public class MouseLook : MonoBehaviour
{
    public InputActionReference horizontalLook;
    public InputActionReference verticalLook;
    public InputActionReference upAction;
    public InputActionReference leftAction;
    public InputActionReference downAction;
    public InputActionReference rightAction;

    public float lookspeed = 1f;
    public float moveSpeed = 5f;
    public Transform cameraTransform;
    public float maxPitch = 80f;

    private float pitch;
    private float yaw;

    private Vector2 movementInput;
    private bool controlsEnabled = false;

    void Start()
    {
        Cursor.lockState = CursorLockMode.Locked;

        horizontalLook.action.performed += HandleHorizontalLookChange;
        verticalLook.action.performed += HandleVerticalLookChange;

        upAction.action.performed += _ => StartMovement(Vector2.up);
        leftAction.action.performed += _ => StartMovement(Vector2.left);
        downAction.action.performed += _ => StartMovement(Vector2.down);
        rightAction.action.performed += _ => StartMovement(Vector2.right);

        upAction.action.canceled += _ => StopMovement();
        leftAction.action.canceled += _ => StopMovement();
        downAction.action.canceled += _ => StopMovement();
        rightAction.action.canceled += _ => StopMovement();

        horizontalLook.action.Enable();
        verticalLook.action.Enable();
        upAction.action.Enable();
        leftAction.action.Enable();
        downAction.action.Enable();
        rightAction.action.Enable();
    }

    void OnDisable()
    {
        // Make sure to disable actions when the script is disabled or destroyed
        horizontalLook.action.Disable();
        verticalLook.action.Disable();
        upAction.action.Disable();
        leftAction.action.Disable();
        downAction.action.Disable();
        rightAction.action.Disable();
    }

    void HandleHorizontalLookChange(InputAction.CallbackContext obj)
    {
        if (controlsEnabled)
        {
            yaw += obj.ReadValue<float>() * lookspeed * Time.deltaTime;
            transform.localRotation = Quaternion.Euler(0f, yaw, 0f);
        }
    }

    void HandleVerticalLookChange(InputAction.CallbackContext obj)
    {
        if (controlsEnabled)
        {
            pitch -= obj.ReadValue<float>() * lookspeed * Time.deltaTime;
            pitch = Mathf.Clamp(pitch, -maxPitch, maxPitch);
            cameraTransform.localRotation = Quaternion.Euler(pitch, 0f, 0f);
        }
    }

    void StartMovement(Vector2 direction)
    {
        if (controlsEnabled)
        {
            movementInput = direction;
        }
    }

    void StopMovement()
    {
        if (controlsEnabled)
        {
            movementInput = Vector2.zero;
        }
    }

    void Update()
    {
        if (controlsEnabled)
        {
            Vector3 movementDirection = new Vector3(movementInput.x, 0f, movementInput.y);
            movementDirection = transform.TransformDirection(movementDirection);

            // Keep the movement in the horizontal plane
            movementDirection.y = 0;

            transform.position += movementDirection.normalized * moveSpeed * Time.deltaTime;
        }

        // Toggle controls on/off with the F12 key
        if (Keyboard.current.f12Key.wasPressedThisFrame)
        {
            controlsEnabled = !controlsEnabled;

            if (controlsEnabled)
            {
                Cursor.lockState = CursorLockMode.Locked;
            }
            else
            {
                Cursor.lockState = CursorLockMode.None;
            }
        }
    }
}
