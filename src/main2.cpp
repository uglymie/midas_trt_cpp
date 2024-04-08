#include <libwebsockets.h>
#include <opencv2/opencv.hpp>

using namespace cv;

VideoCapture cap("D://video/sample.mp4");
Mat frame;

static int callback(struct lws *wsi, enum lws_callback_reasons reason, void *user, void *in, size_t len)
{

    switch (reason)
    {
    case LWS_CALLBACK_ESTABLISHED:       // 当服务器和客户端完成握手后
        printf("Client connect!\n");
        break;
    case LWS_CALLBACK_RECEIVE:           // 当接收到客户端发来的
        printf("recvied message: %s\n", in);

        // lws_rx_flow_control( wsi, 0 );
        // 需要给客户端应答时，触发一次写回调
        lws_callback_on_writable( wsi );
        break;
    case LWS_CALLBACK_SERVER_WRITEABLE:
    {
        lwsl_user("LWS_CALLBACK_SERVER_WRITEABLE\n");
        // Read the next frame from the video file
        cap >> frame;

        if (frame.empty())
        {
            lwsl_notice("End of video\n");
            return -1; // End of video
        }
        // cv::resize(frame, frame, cv::Size(320, 180));
        cv::resize(frame, frame, cv::Size(640, 320));
        cv::waitKey(25);
        // Convert the frame to JPEG format
        std::vector<uchar> img_buffer;
        imencode(".jpg", frame, img_buffer);
        lwsl_user("frame size : %d", frame.size());
        lwsl_user("img_buffer size : %d", img_buffer.size());
        lwsl_user("Sending JPEG data to the client\n");

        unsigned char *buf = (unsigned char *)malloc(LWS_PRE + img_buffer.size() + LWS_SEND_BUFFER_POST_PADDING);
        memcpy(&buf[LWS_PRE], img_buffer.data(), img_buffer.size());
        // Send the JPEG data to the client
        // lws_write(wsi, img_buffer.data(), img_buffer.size(), LWS_WRITE_BINARY);
        lws_write(wsi, &buf[LWS_PRE], img_buffer.size(), LWS_WRITE_BINARY);
        
        // const std::string message = "Hello, WebSockets!";
        // unsigned char *buf = (unsigned char *)malloc(LWS_PRE + message.size() + LWS_SEND_BUFFER_POST_PADDING);
        // // Copy the message to the buffer, starting from LWS_PRE
        // memcpy(&buf[LWS_PRE], message.c_str(), message.size());
        // lws_write(wsi, &buf[LWS_PRE], message.size(), LWS_WRITE_TEXT);

        // Free the allocated memory
        free(buf);

        // Request another LWS_CALLBACK_SERVER_WRITEABLE event
        lws_callback_on_writable(wsi);
        // 下面的调用允许在此连接上接收数据
        // lws_rx_flow_control( wsi, 1 );

        break;
    }

    default:
        break;
    }

    return 0;
}

static struct lws_protocols protocols[] = {
    {"http", lws_callback_http_dummy, 0, 0, 0, NULL, 0},
    {"ws", callback, 0, 0,},
    {NULL, NULL, 0, 0} /* 最后一个协议必须是 NULL 结束的 */
};

static const struct lws_http_mount mount = {
    /* .mount_next */ NULL,         /* linked-list "next" */
    /* .mountpoint */ "/",          /* mountpoint URL */
    /* .origin */ "./video-stream", /* serve from dir */
    /* .def */ "index.html",        /* default filename */
    /* .protocol */ NULL,
    /* .cgienv */ NULL,
    /* .extra_mimetypes */ NULL,
    /* .interpret */ NULL,
    /* .cgi_timeout */ 0,
    /* .cache_max_age */ 0,
    /* .auth_mask */ 0,
    /* .cache_reusable */ 0,
    /* .cache_revalidate */ 0,
    /* .cache_intermediaries */ 0,
    /* .origin_protocol */ LWSMPRO_FILE, /* files in a dir */
    /* .mountpoint_len */ 1,             /* char count */
    /* .basic_auth_login_file */ NULL,
};

int main(void)
{
    struct lws_context_creation_info info;
    memset(&info, 0, sizeof info);

    struct lws_context *context;

    int n = 0, logs = LLL_USER | LLL_ERR | LLL_WARN | LLL_NOTICE;

    lws_set_log_level(logs, NULL);
    lwsl_user("LWS minimal ws server | visit http://localhost:9002 (-s = use TLS / https)\n");

    info.vhost_name = "168.168.0.17";
    info.port = 9002;
    info.iface = NULL;
    info.gid = -1;
    info.uid = -1;
    info.protocols = protocols;
    info.mounts = &mount;
    info.extensions = NULL;
    info.options = LWS_SERVER_OPTION_VALIDATE_UTF8;

    context = lws_create_context(&info);
    if (!context)
    {
        fprintf(stderr, "libwebsocket init failed\n");
        return -1;
    }

    printf("WebSocket server started. Listening on port %d...\n", info.port);

    while (1)
    {
        lws_service(context, 50); // 50 ms timeout for event handling
    }

    lws_context_destroy(context);

    return 0;
}
