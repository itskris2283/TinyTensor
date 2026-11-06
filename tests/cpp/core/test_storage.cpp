#include <gtest/gtest.h>
#include "tinytensor/core/storage.hpp"
#include "tinytensor/utils/errors.hpp"

using namespace tinytensor;

TEST(StorageTest, AlignedAllocation) {
    Storage s(1024);
    EXPECT_NE(s.data(), nullptr);
    EXPECT_EQ(s.nbytes(), 1024);
    EXPECT_TRUE(s.is_owning());
    uintptr_t addr = reinterpret_cast<uintptr_t>(s.data());
    EXPECT_EQ(addr % MEMORY_ALIGNMENT, 0u);
}

TEST(StorageTest, ZeroBytesAllocation) {
    Storage s(0);
    EXPECT_EQ(s.data(), nullptr);
    EXPECT_EQ(s.nbytes(), 0);
}

TEST(StorageTest, ExternalNonOwningStorage) {
    alignas(64) float buffer[16] = {0};
    Storage s(buffer, sizeof(buffer), false);
    EXPECT_EQ(s.data(), buffer);
    EXPECT_FALSE(s.is_owning());
}

TEST(StorageTest, MoveSemantics) {
    Storage s1(512);
    void* ptr = s1.data();
    Storage s2 = std::move(s1);
    EXPECT_EQ(s2.data(), ptr);
    EXPECT_EQ(s2.nbytes(), 512);
    EXPECT_EQ(s1.data(), nullptr);
    EXPECT_EQ(s1.nbytes(), 0);
}

TEST(StorageTest, ExceedsResourceLimit) {
    EXPECT_THROW(Storage(MAX_ALLOCATION_BYTES + 1), ResourceLimitError);
}
