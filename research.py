import cv2
import time

from TT_ball_speed.ball_speed import BallSpeed, Debug


def run_vid():
    vid = cv2.VideoCapture('data/big.mp4')
    fps = vid.get(cv2.CAP_PROP_FPS)
    ball_speed = BallSpeed(fps, debug=Debug.NONE)

    while True:
        start_time = time.time()
        success, img = vid.read()
        if not success:
            break
        speed = ball_speed.process(img)
        imS = cv2.resize(img, (540, 960))
        cv2.putText(imS, f'{str(speed)} RPS', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Vid", imS)

        if cv2.waitKey(15) & 0xFF == ord('q'):
            ball_speed.debug_plot_speed()
            break

        end_time = time.time()
        # print((end_time - start_time))

    cv2.destroyAllWindows()
    ball_speed.debug_plot_speed()


run_vid()

