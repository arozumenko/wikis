#import <Foundation/Foundation.h>

@interface Greeter : NSObject
- (NSString *)greet:(NSString *)name;
- (NSString *)formatName:(NSString *)n;
@end

@implementation Greeter
- (NSString *)greet:(NSString *)name {
    return [self formatName:name];
}
- (NSString *)formatName:(NSString *)n {
    return [n uppercaseString];
}
@end
