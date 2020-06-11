#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <opencv2/core.hpp>


struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
};

//
// Websites which have been used as a reference for implementing this ringbuffer:
// https://stackoverflow.com/questions/8054273/how-to-implement-an-stl-style-iterator-and-avoid-common-pitfalls
// https://stackoverflow.com/questions/7758580/writing-your-own-stl-container/7759622#7759622
// https://www.ibm.com/support/knowledgecenter/ssw_ibm_i_74/rzarg/cplr330.htm
//

// Ringbuffer class.
// T is the data type, and Capacity is the capacity of the ringbuffer.
template<typename T, int Capacity>
class RingBuffer {

    // The actual data.
    T *data_[Capacity];
    // The capacity of the ringbuffer -- stored as a variable for easy access.
    const int capacity_{Capacity};
    // Current number of entries in the ringbuffer.
    int size_{0};
    // Index of the first element in the buffer.
    int first_{0};

    public:
    // Constructor
    RingBuffer() {
        for (int ii=0; ii<capacity_; ii++) {
            data_[ii] = nullptr;
        }
    }
    // Destructor
    virtual ~RingBuffer() {
        // Free all remaining entries.
        for (int ii=0; ii<capacity_; ii++) {
            if (data_[ii] != nullptr) {
                std::cout << "Deleting remaining entry with value " << (*data_[ii]) << "\n";
                delete data_[ii];
                data_[ii] = nullptr;
            }
        }
    }

    // Non-const iterator for this ringbuffer (not a complete implementation as required by the C++ standard).
    class iterator {
        // Type of the associated ringbuffer.
        using RingBufferType = RingBuffer<T, Capacity>;
        // Pointer to the associated ringbuffer object.
        RingBufferType *rb_;
        // Index in the ringbuffer where this iterator is pointing to.
        // This is not the actual index in the data_ member of the ringbuffer.
        int index_;

        public:
        // constructor
        iterator(RingBufferType *rb, int idx) : rb_(rb), index_(idx) {

        }

        // Prefix increment
        iterator& operator++() {
            // index_ = (index_ + 1) % rb_->capacity_;
            index_++;
            return *this;
        }

        // Postfix increment
        iterator operator++(int) {
            // index_ = (index_ + 1) % rb_->capacity_;
            index_++;
            return *this;
        }

        // Check for equality
        bool operator==(const iterator& it) { return this->index_ == it.index_; }
        bool operator!=(const iterator& it) { return !(this->operator==(it)); }

        // Access to reference.
        T& operator*() {
            // return *(rb_->data_[index_]);
            return *(rb_->data_[(rb_->first_ + index_) % rb_->capacity_]);
        }

        // Access to pointer.
        T* operator->() {
            // return rb_->data_[index_];
            return rb_->data_[(rb_->first_ + index_) % rb_->capacity_];
        }
    };

    // Add an element to the end.
    // If the ring buffer is full, the oldest element is deleted.
    void push_back(const T& new_entry) {
        // First check if the buffer is full.
        if (size_ == capacity_) {
            pop_front();
        }

        // Index where the next element is to be put.
        int next = (first_ + size_) % capacity_;
        data_[next] = new T(new_entry);
        size_++;

        std::cout << "Added new entry with value " << new_entry << " in ringbuffer at position " << next << "\n";
        std::cout << "Size is now " << size_ << " and first is " << first_ << "\n";
    }

    // Remove the oldest element from the front.
    void pop_front() {
        // Make sure there is a valid entry at the front
        if (data_[first_] == nullptr) {
            return;
        }
        // Pop it away.
        std::cout << "Deleting entry at position " << first_ << " with value " << (*data_[first_]) << "\n";
        delete data_[first_];
        data_[first_] = nullptr;
        first_ = (first_ + 1) % capacity_;
        size_--;
        std::cout << "Size is now " << size_ << " and first is " << first_ << "\n";
    }

    iterator begin() {
        // iterator it(this, first_);
        iterator it(this, 0);
        return it;
    }

    iterator end() {
        // iterator it(this, (first_ + size_) % capacity_);
        iterator it(this, size_);
        return it;
    }

    // Access operator.
    T& operator[](int idx) { return *data_[(first_ + idx) % capacity_]; }
    const T& operator[](int idx) const { return *data_[(first_ + idx) % capacity_]; }

    // Get the size
    int size() const { return size_; }
    

};

inline void test_ringbuffer() {
    RingBuffer<double, 5> rb;

    rb.push_back(3.0);
    rb.push_back(8.0);

    std::cout << "Current content of the ringbuffer:\n";
    for (RingBuffer<double, 5>::iterator it = rb.begin(); it != rb.end(); ++it) {
        std::cout << "* " << (*it) << "\n";
    }

    rb.pop_front();
    rb.push_back(12.0);
    rb.push_back(-12.0);
    rb.push_back(-1.0);
    rb.push_back(3.14);

    std::cout << "Current content of the ringbuffer:\n";
    for (double d : rb) {
        std::cout << "* " << d << "\n";
    }

    rb.push_back(20.0);
    rb.push_back(30.0);
    rb.push_back(40.0);
    rb.push_back(50);

    std::cout << "Ringbuffer size = " << rb.size() << "\n";

    std::cout << "Current content of the ringbuffer:\n";
    for (double d : rb) {
        std::cout << "* " << d << "\n";
    }
    std::cout << "Current content of the ringbuffer:\n";
    for (int ii=0; ii<rb.size(); ii++) {
        std::cout << "* " << rb[ii] << "\n";
    }

    rb.pop_front();
    rb.pop_front();
    std::cout << "Ringbuffer size = " << rb.size() << "\n";

}


#endif /* dataStructures_h */
