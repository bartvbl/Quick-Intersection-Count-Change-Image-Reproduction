#include <algorithm>
#include <sstream>
#include "Histogram.h"

void Histogram::ensureKeyExists(unsigned int key, std::map<unsigned int, size_t> &map) const {
    auto it = map.find(key);
    if(it == map.end()) {
        map[key] = 0;
    }
}

Histogram Histogram::merge(Histogram other) {
    Histogram mergedHistogram;

    // Dump elements from this histogram first
    for (auto &content : contents) {
        mergedHistogram.contents[content.first] = content.second;
    }

    for (auto &content : other.contents) {
        ensureKeyExists(content.first, mergedHistogram.contents);
        mergedHistogram.contents[content.first] += content.second;
    }

    return mergedHistogram;
}

void Histogram::count(size_t key) {
    ensureKeyExists(key, contents);
    contents[key]++;
}

std::string Histogram::toJSON(int indentLevel) {
    std::vector<unsigned int> keys;
    for (auto &content : contents) {
        keys.push_back(content.first);
    }

    std::sort(keys.begin(), keys.end());

    std::stringstream ss;


    for(int i = 0; i < indentLevel; i++) {
        ss << "\t";
    }
    ss << "{" << std::endl;
    int index = 0;
    for(int i = 0; i < indentLevel + 1; i++) {
        ss << "\t";
    }
    for (auto &key : keys) {
        ss << "\"" << key << "\": " << contents[key] << (index == keys.size() - 1 ? "" : ",") << "\t";
        index++;
        if(index > 0 && index % 10 == 0) {
            ss << std::endl;
            for(int i = 0; i < indentLevel + 1; i++) {
                ss << "\t";
            }
        }
    }
    ss << std::endl;
    for(int i = 0; i < indentLevel; i++) {
        ss << "\t";
    }
    ss << "}";

    return ss.str();
}

std::map<unsigned int, size_t> Histogram::getMap() {
    return contents;
}
