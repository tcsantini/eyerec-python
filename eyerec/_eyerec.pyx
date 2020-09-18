from typing import Any, Dict

cimport numpy as np

from libcpp cimport bool


cdef extern from "<opencv2/core.hpp>":
    int CV_8UC1


cdef extern from "<opencv2/core.hpp>" namespace "cv":
    cdef cppclass Mat :
        Mat() except +
        Mat(int height, int width, int type, void* data) except +
        unsigned char* data

    cdef cppclass Point_[_Tp]:
        Point_() except +
        _Tp x
        _Tp y

    cdef cppclass Size_[_Tp]:
        Size_() except +
        _Tp height
        _Tp width

    cdef cppclass Rect_[_Tp]:
        Rect_(_Tp _x, _Tp y, _Tp width, _Tp height) except +
        _Tp height
        _Tp width
        _Tp x
        _Tp y

    ctypedef Point_[float] Point2f
    ctypedef Size_[float] Size2f
    ctypedef Rect_[int] Rect


ctypedef double Timestamp


cdef extern from "eyerec/Pupil.hpp":

    cdef struct Pupil:
        Point2f center
        Size2f size
        double angle
        double confidence


cdef extern from "eyerec/PupilTrackingMethod.hpp":

    cdef struct TrackingParameters:
        Rect roi
        float userMinPupilDiameterPx
        float userMaxPupilDiameterPx
        bool provideConfidence
        Timestamp maxAge
        float minDetectionConfidence

    cdef cppclass PupilTrackingMethod:
        Pupil detectAndTrack(const Timestamp& ts, const Mat& frame, TrackingParameters params)


cdef extern from "eyerec/PuReST.hpp":
    cdef cppclass PuReST:
        Pupil detectAndTrack(const Timestamp& ts, const Mat& frame, TrackingParameters params)


cdef extern from "eyerec/TrackingByDetection.hpp":
    cdef cppclass TrackingByDetection[T]:
        Pupil detectAndTrack(const Timestamp& ts, const Mat& frame, TrackingParameters params)


cdef extern from "eyerec/PupilDetectionMethod.hpp":
    cdef cppclass PuRe:
        pass


cdef class PupilTracker:
    cdef TrackingParameters params
    cdef PupilTrackingMethod* pupil_tracking_method

    def __cinit__(self, name: str, *args, **kwargs):
        name = name.lower()
        if name == "purest":
            self.pupil_tracking_method = <PupilTrackingMethod *> new PuReST()
        elif name == "pure":
            self.pupil_tracking_method = <PupilTrackingMethod *> new TrackingByDetection[PuRe]()
        else:
            raise ValueError(f"Unexpected pupil tracker name: {name}")

    def __dealloc__(self):
        del self.pupil_tracking_method

    cdef _detect(self, timestamp: float, frame: np.ndarray):
        cdef unsigned char[:, ::1] gray = frame
        cdef Mat mat_frame = Mat(frame.shape[0], frame.shape[1], CV_8UC1, <void *> &gray[0, 0])
        cdef Pupil pupil = self.pupil_tracking_method.detectAndTrack(timestamp, mat_frame, self.params)
        # it seems cython can't convert the struct with classes to a dict automatically
        return {
                "center": (pupil.center.x, pupil.center.y),
                "size": (pupil.size.width, pupil.size.height),
                "angle": pupil.angle,
                "confidence": pupil.confidence,
                }

    def detect(self, timestamp: float, frame: np.ndarray) -> Dict[str, Any]:
        return self._detect(timestamp, frame)

    def set_roi(self, x: int, y: int, width: int, height: int):
        self.params.roi = Rect(x, y, width, height)

    def set_min_pupil_diameter_px(self, diameter: float):
        self.params.userMinPupilDiameterPx = diameter

    def set_max_pupil_diameter_px(self, diameter: float):
        self.params.userMaxPupilDiameterPx = diameter
