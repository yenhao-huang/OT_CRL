# Variables
PINPATH_3_30 := /home/user/Desktop/oss-arch-gym/sims/DRAM/cpusim/pin
CXX := g++
CXXFLAGS := -shared -fPIC -I$(PINPATH_3_30)/source/include/pin -I$(PINPATH_3_30)/source/include/pin/gen
CXXFLAGS += -I$(PINPATH_3_30)/extras/xed-intel64/include/xed
CXXFLAGS += -I$(PINPATH_3_30)/extras/crt/include
LDFLAGS := -L$(PINPATH_3_30)/intel64/lib -L$(PINPATH_3_30)/intel64/lib-ext -lpin -lxed $(PINPATH_3_30)/intel64/runtime/pincrt/crtendS.o -lpindwarf  -ldl-dynamic -nostdlib  -lm-dynamic -lc-dynamic -lunwind-dynamic
TARGET := CannyMemTrace.so
SOURCES := CannyFaultInjector.cpp
OBJECTS := $(SOURCES:.cpp=.o)

# Rules
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

clean:
	rm -f $(TARGET) $(OBJECTS)

.PHONY: all clean
