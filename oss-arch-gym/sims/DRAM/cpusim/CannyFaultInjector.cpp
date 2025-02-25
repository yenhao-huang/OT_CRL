#include <pin.H>
#include <iostream>
#include <fstream>

// log vairable
std::ofstream TraceFile;

// A flag to detect if we are inside Canny function
bool insideCanny = false;

// Function to log memory accesses
VOID RecordMemoryAccess(VOID* ip, VOID* address, std::string accessType) {
    if (insideCanny) {
        TraceFile << accessType << " at " << address << " by instruction at " << ip << std::endl;
    }
}

// Callbacks for memory reads and writes
VOID MemoryRead(VOID* ip, VOID* addr) {
    RecordMemoryAccess(ip, addr, "READ");
}

VOID MemoryWrite(VOID* ip, VOID* addr) {
    RecordMemoryAccess(ip, addr, "WRITE");
}

// Instrumentation function
VOID Instruction(INS ins, VOID* v) {
    if (INS_IsMemoryRead(ins)) {
        INS_InsertPredicatedCall(
            ins, IPOINT_BEFORE, (AFUNPTR)MemoryRead,
            IARG_INST_PTR,
            IARG_MEMORYREAD_EA,
            IARG_END);
    }

    if (INS_IsMemoryWrite(ins)) {
        INS_InsertPredicatedCall(
            ins, IPOINT_BEFORE, (AFUNPTR)MemoryWrite,
            IARG_INST_PTR,
            IARG_MEMORYWRITE_EA,
            IARG_END);
    }
}

// Function called at the start and end of Canny
VOID BeforeCanny() {
    insideCanny = true;
}

VOID AfterCanny() {
    insideCanny = false;
}

// This function is called for every routine loaded
VOID Routine(RTN rtn, VOID* v) {
    if (RTN_Name(rtn) == "Canny") {
        RTN_Open(rtn);
        RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR)BeforeCanny, IARG_END);
        RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR)AfterCanny, IARG_END);
        RTN_Close(rtn);
    }
}

// This function is called when the application exits
VOID Fini(INT32 code, VOID* v) {
    TraceFile << "Terminating trace..." << std::endl;
    TraceFile.close();
}

int main(int argc, char* argv[]) {
    PIN_Init(argc, argv);

    TraceFile.open("memtrace.out");

    RTN_AddInstrumentFunction(Routine, NULL);
    INS_AddInstrumentFunction(Instruction, NULL);
    PIN_AddFiniFunction(Fini, NULL);

    PIN_StartProgram();  // Start the program.
    return 0;
}
