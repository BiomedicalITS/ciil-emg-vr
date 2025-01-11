using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class TimerController : MonoBehaviour
{
    public TextMeshProUGUI timerText;
    public float initialTime = 300f; // Initial time in seconds
    public OnlyOneGrabAtATime onlyOneGrabAtATime;
    public TcpClientManager _server;
    private EMGSetUp _emg;
    private string modeLabel;

    private float currentTime;
    private bool isTimerRunning;

    // Events for PositionSaver to subscribe to
    public delegate void TimerEvent();
    public event TimerEvent OnTimerStart;
    public event TimerEvent OnTimerStop;
    public event TimerEvent OnTimerReset;
    public event TimerEvent OnTimerComplete;

    // Property to check if the timer is running
    public bool IsTimerRunning
    {
        get { return isTimerRunning; }
    }

    private void Start()
    {
        _emg = FindObjectOfType<EMGSetUp>();
        currentTime = initialTime;
        isTimerRunning = false;
        UpdateTimerText();
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            if (isTimerRunning)
            {
                StopTimer();
            }
            else
            {
                StartTimer();
                onlyOneGrabAtATime.trialtime = 0f;
            }
        }
        else if (Input.GetKeyDown(KeyCode.R))
        {
            ResetTimer();
        }

        if (isTimerRunning)
        {
            currentTime -= Time.deltaTime;
            if (currentTime <= 0f)
            {
                currentTime = 0f;
                StopTimer();
                if (OnTimerComplete != null)
                {
                    modeLabel = PlayerPrefs.GetString("Mode");
                    OnTimerComplete.Invoke();
                }
            }
            UpdateTimerText();
        }
    }

    private void UpdateTimerText()
    {
        int minutes = Mathf.FloorToInt(currentTime / 60f);
        int seconds = Mathf.FloorToInt(currentTime % 60f);
        string timeString = string.Format("{0:00}:{1:00}", minutes, seconds);
        timerText.text = timeString;
    }

    public void StartTimer()
    {
        modeLabel = PlayerPrefs.GetString("Mode");
        isTimerRunning = true;
        OnTimerStart?.Invoke();
    }

    public void StopTimer()
    {
        isTimerRunning = false;
        OnTimerStop?.Invoke();
    }

    public void ResetTimer()
    {
        isTimerRunning = false;
        currentTime = initialTime;
        UpdateTimerText();
        modeLabel = PlayerPrefs.GetString("Mode");
        OnTimerReset?.Invoke();
    }
}
