#ifndef MUSIC_BEAT_DETECTOR_UTILS_CLASS_MACRO_H
#define MUSIC_BEAT_DETECTOR_UTILS_CLASS_MACRO_H

#define DECLARE_NOT_COPYABLE(className)                                                                                \
    className(const className&) = delete;                                                                              \
    className& operator=(const className&) = delete

#define DECLARE_NOT_MOVABLE(className)                                                                                 \
    className(className&&) = delete;                                                                                   \
    className& operator=(className&&) = delete

#endif
