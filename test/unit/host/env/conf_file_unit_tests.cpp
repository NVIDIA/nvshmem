#include <gtest/gtest.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <sys/stat.h>
#include <unistd.h>

#include "bootstrap_host_transport/env_defs_internal.h"

static std::string writeFile(const std::string &path, const std::string &contents) {
    FILE *fp = std::fopen(path.c_str(), "w");
    EXPECT_NE(fp, nullptr) << "Failed to open file: " << path << " errno=" << errno;
    if (fp == nullptr) return path;
    std::fwrite(contents.data(), 1, contents.size(), fp);
    std::fclose(fp);
    return path;
}

static std::string makeTempDir() {
    char tmpTemplate[] = "/tmp/nvshmem_conf_test.XXXXXX";
    char *tmpDir = mkdtemp(tmpTemplate);
    EXPECT_NE(tmpDir, nullptr);
    return tmpDir ? std::string(tmpDir) : std::string("/tmp");
}

class EnvConfFileTest : public ::testing::Test {
   protected:
    void SetUp() override {
        nvshmemi_conf_reset_cache();
        unsetenv("NVSHMEM_CONF_FILE");
        unsetenv("HOME");
        unsetenv("NVSHMEM_UNITTEST_DEBUG");
        unsetenv("NVSHMEM_UNITTEST_SOME_INT");
        unsetenv("NVSHMEM_UNITTEST_FOO");
    }

    void TearDown() override { nvshmemi_conf_reset_cache(); }
};

TEST_F(EnvConfFileTest, ConfOverridesEnvAndRespectsOrder_UserThenSpecified) {
    std::string homeDir = makeTempDir();
    std::string userConf = homeDir + "/.nvshmem.conf";
    std::string specifiedConf = homeDir + "/specified.conf";

    (void)writeFile(userConf,
                    std::string("# user config\n") +
                        "NVSHMEM_UNITTEST_DEBUG=INFO\n" +
                        "NVSHMEM_UNITTEST_SOME_INT =  7  \n");

    (void)writeFile(specifiedConf,
                    std::string("# specified config\n") +
                        "NVSHMEM_UNITTEST_DEBUG=WARN\n" +
                        "NVSHMEM_UNITTEST_SOME_INT=42 # trailing comment\n" +
                        "# NVSHMEM_IGNORED=1\n");

    ASSERT_EQ(setenv("HOME", homeDir.c_str(), 1), 0);
    ASSERT_EQ(setenv("NVSHMEM_CONF_FILE", specifiedConf.c_str(), 1), 0);

    /* Environment vars should lose to config if config provides the key. */
    ASSERT_EQ(setenv("NVSHMEM_UNITTEST_DEBUG", "ERROR", 1), 0);
    ASSERT_EQ(setenv("NVSHMEM_UNITTEST_SOME_INT", "99", 1), 0);

    const char *debug = nvshmemi_getenv("UNITTEST_DEBUG");
    ASSERT_NE(debug, nullptr);
    EXPECT_STREQ(debug, "WARN");

    const char *someInt = nvshmemi_getenv("UNITTEST_SOME_INT");
    ASSERT_NE(someInt, nullptr);
    EXPECT_STREQ(someInt, "42");
}

TEST_F(EnvConfFileTest, ConfMissingKeyFallsBackToEnv) {
    std::string homeDir = makeTempDir();
    std::string specifiedConf = homeDir + "/specified.conf";

    (void)writeFile(specifiedConf,
                    std::string("# no NVSHMEM_UNITTEST_FOO here\n") + "NVSHMEM_BAR=1\n");

    ASSERT_EQ(setenv("HOME", homeDir.c_str(), 1), 0);
    ASSERT_EQ(setenv("NVSHMEM_CONF_FILE", specifiedConf.c_str(), 1), 0);
    ASSERT_EQ(setenv("NVSHMEM_UNITTEST_FOO", "from_env", 1), 0);

    const char *foo = nvshmemi_getenv("UNITTEST_FOO");
    ASSERT_NE(foo, nullptr);
    EXPECT_STREQ(foo, "from_env");
}

TEST_F(EnvConfFileTest, MissingAndUnreadableFilesAreIgnored) {
    std::string dir = makeTempDir();
    std::string unreadableConf = dir + "/unreadable.conf";

    /* Force HOME to a clean temp dir so ~/.nvshmem.conf can't influence the result. */
    ASSERT_EQ(setenv("HOME", dir.c_str(), 1), 0);

    (void)writeFile(unreadableConf, "NVSHMEM_UNITTEST_DEBUG=CONF\n");
    ASSERT_EQ(chmod(unreadableConf.c_str(), 0000), 0);

    ASSERT_EQ(setenv("NVSHMEM_CONF_FILE", unreadableConf.c_str(), 1), 0);

    /* Should fall back to env because the specified conf is unreadable and /etc shouldn't
     * contain our unit-test-only key. */
    ASSERT_EQ(setenv("NVSHMEM_UNITTEST_DEBUG", "FROM_ENV", 1), 0);

    const char *debug = nvshmemi_getenv("UNITTEST_DEBUG");
    ASSERT_NE(debug, nullptr);
    EXPECT_STREQ(debug, "FROM_ENV");

    /* Best-effort cleanup. */
    (void)chmod(unreadableConf.c_str(), 0600);
}

TEST_F(EnvConfFileTest, ParsingEdgeCases) {
    std::string dir = makeTempDir();
    std::string conf = dir + "/edge.conf";

    (void)writeFile(conf,
                    std::string("  # leading whitespace comment\r\n") +
                        "NVSHMEM_A=1\r\n"
                        "NVSHMEM_EMPTY=\n"
                        "NVSHMEM_WS =  2  \n"
                        "NOEQUALS\n"
                        "NVSHMEM_HASH=va#lue\n"
                        "NVSHMEM_HASH2=va #lue\n"
                        "NVSHMEM_HASH3=#value\n");

    ASSERT_EQ(setenv("NVSHMEM_CONF_FILE", conf.c_str(), 1), 0);

    const char *a = nvshmemi_getenv("A");
    ASSERT_NE(a, nullptr);
    EXPECT_STREQ(a, "1");

    const char *empty = nvshmemi_getenv("EMPTY");
    ASSERT_NE(empty, nullptr);
    EXPECT_STREQ(empty, "");

    const char *ws = nvshmemi_getenv("WS");
    ASSERT_NE(ws, nullptr);
    EXPECT_STREQ(ws, "2");

    const char *hash = nvshmemi_getenv("HASH");
    ASSERT_NE(hash, nullptr);
    EXPECT_STREQ(hash, "va#lue");

    const char *hash2 = nvshmemi_getenv("HASH2");
    ASSERT_NE(hash2, nullptr);
    EXPECT_STREQ(hash2, "va");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
