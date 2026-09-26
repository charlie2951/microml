USERMOD_DIR := $(USERMOD_DIR)

# Add C files
SRC_USERMOD += $(USERMOD_DIR)/microml.c
SRC_USERMOD += $(USERMOD_DIR)/mlp.c
SRC_USERMOD += $(USERMOD_DIR)/cnn.c
#SRC_USERMOD += $(USERMOD_DIR)/svm.c

# Add flags if required
CFLAGS_USERMOD += -I$(USERMOD_DIR)
