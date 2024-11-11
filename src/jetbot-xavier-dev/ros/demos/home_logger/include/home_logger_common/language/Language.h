#ifndef HOME_LOGGER_COMMON_LANGUAGE_LANGUAGE_H
#define HOME_LOGGER_COMMON_LANGUAGE_LANGUAGE_H

#include <string>

enum class Language
{
    CHINESE,
    ENGLISH,
    FRENCH
};

inline bool languageFromString(const std::string& str, Language& language)
{
    if (str == "zh")
    {
        language = Language::CHINESE;
    }
    else if (str == "en")
    {
        language = Language::ENGLISH;
    }
    else if (str == "fr")
    {
        language = Language::FRENCH;
    }
    else
    {
        return false;
    }

    return true;
}

#endif
